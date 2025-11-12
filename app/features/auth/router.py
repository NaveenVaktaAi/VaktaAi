from fastapi import APIRouter, HTTPException, Depends, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import logging
from bson import ObjectId

from app.features.auth.repository import AuthRepository
from app.features.auth.schema import (
    UserSignupRequest, UserLoginRequest, OTPVerificationRequest, ResendOTPRequest,
    EmailVerificationRequest, EmailVerificationConfirmRequest,
    UserUpdateRequest, StudentUpdateRequest,
    SignupResponse, LoginResponse, OTPResponse, AuthResponse, EmailVerificationResponse,
    UserResponse, StudentResponse, UserWithStudentResponse
)
from app.features.auth.utils import jwt_utils, validation_utils
from app.common.schemas import ResponseModal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


# ===== HELPER FUNCTIONS =====

def convert_mongo_doc_to_dict(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB document to proper format for Pydantic models"""
    if not doc:
        return doc
    
    # Convert ObjectId to string
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    
    # Convert other ObjectId fields to string
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            doc[key] = str(value)
    
    return doc


# ===== DEPENDENCY FUNCTIONS =====

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current authenticated user from JWT token"""
    try:
        token = credentials.credentials
        payload = jwt_utils.verify_token(token, "access")
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        auth_repo = AuthRepository()
        user = await auth_repo.get_user_by_id(payload["user_id"])
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Convert ObjectId to string for _id field
        if "_id" in user and isinstance(user["_id"], ObjectId):
            user["_id"] = str(user["_id"])
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ===== AUTHENTICATION ENDPOINTS =====

@router.post("/signup", response_model=SignupResponse)
async def signup(user_data: UserSignupRequest):
    """
    User signup endpoint
    
    Creates a new user account with the provided information.
    For students, also creates a student profile.
    """
    try:
        auth_repo = AuthRepository()
        
        # Validate phone number
        if not validation_utils.validate_phone_number(user_data.phone_number, user_data.phone_country_code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid phone number format"
            )
        
        # Create user
        result = await auth_repo.create_user(user_data)
        
        if not result["success"]:
            if result.get("error") == "PHONE_EXISTS":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Phone number already registered. Please login instead."
                )
            elif result.get("error") == "EMAIL_EXISTS":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered. Please login instead."
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["message"]
                )
        
        # Get created user
        user = await auth_repo.get_user_by_id(result["data"]["user_id"])
        
        # Send welcome email if email is provided (non-blocking)
        welcome_email_sent = False
        if user_data.email:
            try:
                from app.features.auth.utils import email_service
                
                # Send welcome email
                email_sent = await email_service.send_welcome_email(user_data.email, user_data.full_name)
                
                if email_sent:
                    logger.info(f"Welcome email sent successfully to {user_data.email}")
                    welcome_email_sent = True
                else:
                    logger.warning(f"Failed to send welcome email to {user_data.email}")
                    
            except Exception as e:
                logger.error(f"Error sending welcome email to {user_data.email}: {e}")
                # Don't fail the signup if email fails
        
        # Convert MongoDB document to proper format
        user_dict = convert_mongo_doc_to_dict(user) if user else None
        
        return SignupResponse(
            success=True,
            message="User created successfully",
            data=result["data"],
            user=UserResponse(**user_dict) if user_dict else None,
            verification_email_sent=welcome_email_sent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in signup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during signup"
        )


@router.post("/login", response_model=LoginResponse)
async def login(login_data: UserLoginRequest):
    """
    User login endpoint
    
    Authenticates user with phone number and OTP (or Firebase verification).
    Returns access and refresh tokens on successful authentication.
    
    Note: For Firebase Auth, the OTP is verified on the client side.
    Set firebase_verified flag in request if using Firebase.
    """
    try:
        auth_repo = AuthRepository()
        
        # Validate phone number
        if not validation_utils.validate_phone_number(login_data.phone_number, login_data.phone_country_code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid phone number format"
            )
        
        # Check if this is Firebase verification (client already verified OTP)
        # If Firebase is being used, we trust the client-side verification
        firebase_verified = hasattr(login_data, 'firebase_verified') and getattr(login_data, 'firebase_verified', False)
        
        # If otp is "firebase_verified" or starts with special marker, treat as Firebase auth
        is_firebase_auth = login_data.otp and (login_data.otp == "firebase_verified" or login_data.otp.startswith("fb_"))
        
        # Authenticate user with OTP (or Firebase verification)
        result = await auth_repo.authenticate_user(
            login_data.phone_number,
            login_data.phone_country_code,
            login_data.otp,
            firebase_verified=is_firebase_auth
        )
        
        if not result["success"]:
            if "not found" in result["message"].lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            elif "otp" in result["message"].lower():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=result["message"]
                )
            elif "deactivated" in result["message"].lower():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is deactivated"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=result["message"]
                )
        
        user_data = result["data"]["user"]
        
        # Convert MongoDB document to proper format
        user_dict = convert_mongo_doc_to_dict(user_data) if user_data else None
        
        return LoginResponse(
            success=True,
            message="Login successful",
            data={
                "user_id": result["data"]["user_id"],
                "expires_in": 24 * 60 * 60  # 24 hours in seconds
            },
            access_token=result["data"]["access_token"],
            refresh_token=result["data"]["refresh_token"],
            user=UserResponse(**user_dict) if user_dict else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@router.post("/refresh-token", response_model=AuthResponse)
async def refresh_token(refresh_token: str = Header(..., alias="refresh-token")):
    """
    Refresh access token endpoint
    
    Generates a new access token using a valid refresh token.
    """
    try:
        auth_repo = AuthRepository()
        
        result = await auth_repo.refresh_access_token(refresh_token)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result["message"]
            )
        
        return AuthResponse(
            success=True,
            message="Token refreshed successfully",
            data=result["data"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in refresh_token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh"
        )


# ===== OTP ENDPOINTS =====

@router.post("/send-otp", response_model=OTPResponse)
async def send_otp(otp_request: ResendOTPRequest):
    """
    Send OTP endpoint
    
    Sends an OTP to the provided phone number for verification or login.
    """
    try:
        auth_repo = AuthRepository()
        
        # Validate phone number
        if not validation_utils.validate_phone_number(otp_request.phone_number, otp_request.phone_country_code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid phone number format"
            )
        
        result = await auth_repo.send_otp(otp_request.phone_number, otp_request.phone_country_code)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
        return OTPResponse(
            success=True,
            message="OTP sent successfully",
            data=result["data"],
            otp_sent=True,
            expires_in=result["data"]["expires_in"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in send_otp: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during OTP sending"
        )


@router.post("/verify-otp", response_model=AuthResponse)
async def verify_otp(otp_data: OTPVerificationRequest):
    """
    Verify OTP endpoint
    
    Verifies the OTP sent to the phone number.
    """
    try:
        auth_repo = AuthRepository()
        
        # Validate phone number
        if not validation_utils.validate_phone_number(otp_data.phone_number, otp_data.phone_country_code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid phone number format"
            )
        
        result = await auth_repo.verify_otp(otp_data.phone_number, otp_data.phone_country_code, otp_data.otp)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        
        return AuthResponse(
            success=True,
            message="OTP verified successfully",
            data={"phone_number": f"{otp_data.phone_country_code}{otp_data.phone_number}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verify_otp: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during OTP verification"
        )


@router.post("/resend-otp", response_model=OTPResponse)
async def resend_otp(otp_request: ResendOTPRequest):
    """
    Resend OTP endpoint
    
    Resends an OTP to the provided phone number.
    """
    try:
        auth_repo = AuthRepository()
        
        # Validate phone number
        if not validation_utils.validate_phone_number(otp_request.phone_number, otp_request.phone_country_code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid phone number format"
            )
        
        result = await auth_repo.resend_otp(otp_request.phone_number, otp_request.phone_country_code)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
        return OTPResponse(
            success=True,
            message="OTP resent successfully",
            data=result["data"],
            otp_sent=True,
            expires_in=result["data"]["expires_in"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in resend_otp: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during OTP resending"
        )


# ===== EMAIL VERIFICATION ENDPOINTS =====

@router.post("/send-email-verification", response_model=EmailVerificationResponse)
async def send_email_verification(email_data: EmailVerificationRequest):
    """
    Send email verification endpoint
    
    Sends a verification code to the provided email address.
    """
    try:
        auth_repo = AuthRepository()
        
        result = await auth_repo.send_email_verification(email_data.email)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
        return EmailVerificationResponse(
            success=True,
            message="Verification email sent successfully",
            data=result["data"],
            verification_sent=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in send_email_verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during email verification sending"
        )


@router.post("/verify-email", response_model=AuthResponse)
async def verify_email(verification_data: EmailVerificationConfirmRequest):
    """
    Verify email endpoint
    
    Verifies the email address using the verification code.
    """
    try:
        auth_repo = AuthRepository()
        
        result = await auth_repo.verify_email_code(verification_data.email, verification_data.verification_code)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        
        return AuthResponse(
            success=True,
            message="Email verified successfully",
            data={"email": verification_data.email}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verify_email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during email verification"
        )


# ===== USER PROFILE ENDPOINTS =====

@router.get("/me", response_model=UserWithStudentResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user profile endpoint
    
    Returns the current authenticated user's profile with student details if applicable.
    """
    try:
        auth_repo = AuthRepository()
        
        user_with_student = await auth_repo.get_user_with_student(str(current_user["_id"]))
        
        if not user_with_student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        # Convert MongoDB documents to proper format
        user_dict = convert_mongo_doc_to_dict(user_with_student["user"])
        user_response = UserResponse(**user_dict)
        
        student_response = None
        if user_with_student["student"]:
            student_dict = convert_mongo_doc_to_dict(user_with_student["student"])
            student_response = StudentResponse(**student_dict)
        
        return UserWithStudentResponse(
            user=user_response,
            student=student_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user_profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during profile retrieval"
        )


@router.put("/me", response_model=AuthResponse)
async def update_user_profile(
    update_data: UserUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user profile endpoint
    
    Updates the current user's profile information.
    """
    try:
        auth_repo = AuthRepository()
        
        # Check if email is being changed and if it already exists
        if update_data.email and update_data.email != current_user.get("email"):
            existing_user = await auth_repo.get_user_by_credentials("", "")  # We need a method to check by email only
            # For now, we'll skip this check as we don't have get_user_by_email in repository
        
        result = await auth_repo.update_user(str(current_user["_id"]), update_data)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user profile"
            )
        
        return AuthResponse(
            success=True,
            message="User profile updated successfully",
            data={"user_id": str(current_user["_id"])}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in update_user_profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during profile update"
        )


@router.put("/student-profile", response_model=AuthResponse)
async def update_student_profile(
    update_data: StudentUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update student profile endpoint
    
    Updates the current user's student profile (only for users with student role).
    """
    try:
        auth_repo = AuthRepository()
        
        if current_user["role"] != "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Student profile can only be updated for users with student role"
            )
        
        result = await auth_repo.update_student_profile(str(current_user["_id"]), update_data)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update student profile"
            )
        
        return AuthResponse(
            success=True,
            message="Student profile updated successfully",
            data={"user_id": str(current_user["_id"])}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in update_student_profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during student profile update"
        )


# ===== VERIFICATION ENDPOINTS =====

@router.post("/verify-phone", response_model=AuthResponse)
async def verify_phone_number_endpoint(
    phone_number: str,
    phone_country_code: str = "+91",
    current_user: dict = Depends(get_current_user)
):
    """
    Verify phone number endpoint
    
    Marks the current user's phone number as verified.
    """
    try:
        auth_repo = AuthRepository()
        
        # Verify that the phone number matches the current user
        if f"{phone_country_code}{phone_number}" != f"{current_user.get('phone_country_code', '+91')}{current_user.get('phone_number')}":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Phone number does not match current user"
            )
        
        result = await auth_repo.verify_phone(str(current_user["_id"]))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to verify phone number"
            )
        
        return AuthResponse(
            success=True,
            message="Phone number verified successfully",
            data={"phone_number": f"{phone_country_code}{phone_number}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verify_phone_number_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during phone verification"
        )


@router.post("/verify-email-address", response_model=AuthResponse)
async def verify_email_address_endpoint(
    email: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Verify email address endpoint
    
    Marks the current user's email address as verified.
    """
    try:
        auth_repo = AuthRepository()
        
        # Verify that the email matches the current user
        if email != current_user.get("email"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email does not match current user"
            )
        
        result = await auth_repo.verify_email_address(str(current_user["_id"]))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to verify email address"
            )
        
        return AuthResponse(
            success=True,
            message="Email address verified successfully",
            data={"email": email}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verify_email_address_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during email verification"
        )


# ===== LOGOUT ENDPOINT =====

@router.post("/logout", response_model=AuthResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout endpoint
    
    Logs out the current user (in a real implementation, you might want to blacklist the token).
    """
    try:
        # In a production environment, you would typically:
        # 1. Add the token to a blacklist
        # 2. Update user's last_active_at timestamp
        # 3. Clear any server-side sessions
        
        auth_repo = AuthRepository()
        await auth_repo.update_active_status(str(current_user["_id"]), True)  # Keep user active, just update timestamp
        
        return AuthResponse(
            success=True,
            message="Logged out successfully",
            data={"user_id": str(current_user["_id"])}
        )
        
    except Exception as e:
        logger.error(f"Error in logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )
