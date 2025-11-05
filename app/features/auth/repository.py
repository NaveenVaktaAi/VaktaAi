from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId
from pymongo.database import Database

from app.database.session import get_db
from app.database.mongo_collections import (
    create_user, get_user, get_user_by_phone, get_user_by_email, get_users_by_role,
    update_user, update_user_login, update_user_active_status, delete_user,
    verify_phone_number, verify_email, get_collections,
    create_student, get_student, get_student_by_user_id, get_students_by_class,
    get_students_by_board, get_students_by_exam_target, update_student,
    update_student_by_user_id, delete_student, delete_student_by_user_id
)

from app.features.auth.schema import (
    UserSignupRequest, UserLoginRequest, UserUpdateRequest, StudentUpdateRequest,
    UserResponse, StudentResponse, UserWithStudentResponse, UserRole, AccountStatus,
    CurrentClass, Board, ExamTarget, PreferredLanguage
)

from app.features.auth.utils import (
    password_utils, otp_utils, jwt_utils, referral_utils, validation_utils,
    email_service, sms_utils, cache_utils, OTP_EXPIRE_MINUTES
)


class AuthRepository:
    """Repository for authentication operations"""

    def __init__(self):
        self.db = next(get_db())

    # ===== USER OPERATIONS =====

    async def create_user(self, user_data: UserSignupRequest) -> Dict[str, Any]:
        """Create a new user"""
        try:
            # Check if user already exists
            existing_check = validation_utils.check_user_exists(
                self.db, user_data.phone_number, user_data.email
            )
            
            if existing_check["exists"]:
                if existing_check["by_phone"]:
                    return {"success": False, "message": "Phone number already registered", "error": "PHONE_EXISTS"}
                elif existing_check["by_email"]:
                    return {"success": False, "message": "Email already registered", "error": "EMAIL_EXISTS"}
            
            # Generate referral code
            referral_code = referral_utils.generate_referral_code()
            
            # No password needed - OTP only authentication
            password_hash = None
            
            # Check if user was referred
            referred_by = None
            if user_data.referral_code:
                referrer = self.db["users"].find_one({"referral_code": user_data.referral_code})
                if referrer:
                    referred_by = referrer["_id"]
            
            # Create user document
            user_doc = {
                "phone_number": user_data.phone_number,
                "phone_country_code": user_data.phone_country_code,
                "full_name": user_data.full_name,
                "password_hash": password_hash,
                "is_phone_verified": False,
                "role": user_data.role.value,
                "account_status": AccountStatus.TRIAL.value,
                "is_active": True,
                "email": user_data.email,
                "is_email_verified": False,
                "created_at": datetime.utcnow(),
                "last_login_at": None,
                "last_active_at": None,
                "referral_code": referral_code,
                "referred_by": referred_by,
                "profile_picture_url": None,
                "date_of_birth": user_data.date_of_birth,
                "device_id": None,
                "fcm_token": None,
                "state": user_data.state,
                "city": user_data.city
            }
            
            user_id = create_user(self.db, user_doc)
            
            # Create student profile if role is student
            student_id = None
            if user_data.role == UserRole.STUDENT:
                student_doc = {
                    "user_id": ObjectId(user_id),
                    "current_class": user_data.current_class.value,
                    "board": user_data.board.value,
                    "exam_target": user_data.exam_target.value,
                    "preferred_language": user_data.preferred_language.value,
                    "state": user_data.state,
                    "city": user_data.city,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                student_id = create_student(self.db, student_doc)
            
            return {
                "success": True,
                "message": "User created successfully",
                "data": {
                    "user_id": user_id,
                    "student_id": student_id
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error creating user: {str(e)}", "error": "CREATION_FAILED"}

    async def get_user_by_credentials(self, phone_number: str, phone_country_code: str = "+91") -> Optional[Dict[str, Any]]:
        """Get user by phone number"""
        try:
            return get_user_by_phone(self.db, phone_number, phone_country_code)
        except Exception as e:
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            return get_user(self.db, user_id)
        except Exception as e:
            return None

    async def update_user(self, user_id: str, update_data: UserUpdateRequest) -> bool:
        """Update user information"""
        try:
            update_fields = {}
            
            if update_data.full_name is not None:
                update_fields["full_name"] = update_data.full_name
            if update_data.email is not None:
                update_fields["email"] = update_data.email
                update_fields["is_email_verified"] = False  # Reset verification if email changed
            if update_data.profile_picture_url is not None:
                update_fields["profile_picture_url"] = update_data.profile_picture_url
            if update_data.date_of_birth is not None:
                update_fields["date_of_birth"] = update_data.date_of_birth
            if update_data.state is not None:
                update_fields["state"] = update_data.state
            if update_data.city is not None:
                update_fields["city"] = update_data.city
            
            if update_fields:
                update_user(self.db, user_id, update_fields)
                return True
            return False
            
        except Exception as e:
            return False


    async def verify_phone(self, user_id: str) -> bool:
        """Verify user phone number"""
        try:
            verify_phone_number(self.db, user_id)
            return True
        except Exception as e:
            return False

    async def verify_email_address(self, user_id: str) -> bool:
        """Verify user email address"""
        try:
            verify_email(self.db, user_id)
            return True
        except Exception as e:
            return False

    async def update_login_timestamp(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        try:
            update_user_login(self.db, user_id)
            return True
        except Exception as e:
            return False

    async def update_active_status(self, user_id: str, is_active: bool) -> bool:
        """Update user's active status"""
        try:
            update_user_active_status(self.db, user_id, is_active)
            return True
        except Exception as e:
            return False

    async def delete_user_account(self, user_id: str) -> bool:
        """Delete user account and associated data"""
        try:
            delete_user(self.db, user_id)
            return True
        except Exception as e:
            return False

    # ===== STUDENT OPERATIONS =====

    async def create_student_profile(self, user_id: str, student_data: UserSignupRequest) -> Optional[str]:
        """Create student profile"""
        try:
            student_doc = {
                "user_id": ObjectId(user_id),
                "current_class": student_data.current_class.value,
                "board": student_data.board.value,
                "exam_target": student_data.exam_target.value,
                "preferred_language": student_data.preferred_language.value,
                "state": student_data.state,
                "city": student_data.city,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            return create_student(self.db, student_doc)
            
        except Exception as e:
            return None

    async def get_student_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get student profile by user ID"""
        try:
            return get_student_by_user_id(self.db, user_id)
        except Exception as e:
            return None

    async def update_student_profile(self, user_id: str, update_data: StudentUpdateRequest) -> bool:
        """Update student profile"""
        try:
            update_fields = {}
            
            if update_data.current_class is not None:
                update_fields["current_class"] = update_data.current_class.value
            if update_data.board is not None:
                update_fields["board"] = update_data.board.value
            if update_data.exam_target is not None:
                update_fields["exam_target"] = update_data.exam_target.value
            if update_data.preferred_language is not None:
                update_fields["preferred_language"] = update_data.preferred_language.value
            if update_data.state is not None:
                update_fields["state"] = update_data.state
            if update_data.city is not None:
                update_fields["city"] = update_data.city
            
            if update_fields:
                update_student_by_user_id(self.db, user_id, update_fields)
                return True
            return False
            
        except Exception as e:
            return False

    async def delete_student_profile(self, user_id: str) -> bool:
        """Delete student profile"""
        try:
            delete_student_by_user_id(self.db, user_id)
            return True
        except Exception as e:
            return False

    # ===== OTP OPERATIONS =====

    async def send_otp(self, phone_number: str, phone_country_code: str = "+91") -> Dict[str, Any]:
        """Send OTP to phone number"""
        try:
            # Generate OTP
            otp = otp_utils.generate_otp()
            
            # Store OTP with expiry (in real implementation, use Redis or database)
            otp_expire_time = otp_utils.get_otp_expire_time()
            cache_utils.store_otp(phone_number, otp, OTP_EXPIRE_MINUTES)
            
            # Send OTP via SMS
            sms_sent = await sms_utils.send_otp_sms(phone_number, phone_country_code, otp)
            
            if sms_sent:
                return {
                    "success": True,
                    "message": "OTP sent successfully",
                    "data": {
                        "expires_in": OTP_EXPIRE_MINUTES * 60,  # Convert to seconds
                        "phone_number": f"{phone_country_code}{phone_number}"
                    }
                }
            else:
                return {"success": False, "message": "Failed to send OTP"}
                
        except Exception as e:
            return {"success": False, "message": f"Error sending OTP: {str(e)}"}

    async def verify_otp(self, phone_number: str, phone_country_code: str, otp: str) -> Dict[str, Any]:
        """Verify OTP"""
        try:
            # Get stored OTP
            stored_otp = cache_utils.get_otp(phone_number)
            
            if not stored_otp:
                return {"success": False, "message": "OTP not found or expired"}
            
            if stored_otp != otp:
                return {"success": False, "message": "Invalid OTP"}
            
            # OTP is valid, delete it from cache
            cache_utils.delete_otp(phone_number)
            
            return {"success": True, "message": "OTP verified successfully"}
            
        except Exception as e:
            return {"success": False, "message": f"Error verifying OTP: {str(e)}"}

    async def resend_otp(self, phone_number: str, phone_country_code: str = "+91") -> Dict[str, Any]:
        """Resend OTP"""
        return await self.send_otp(phone_number, phone_country_code)

    # ===== EMAIL VERIFICATION OPERATIONS =====

    async def send_email_verification(self, email: str) -> Dict[str, Any]:
        """Send email verification code"""
        try:
            # Generate verification code
            verification_code = otp_utils.generate_verification_code()
            
            # Store verification code
            cache_utils.store_verification_code(email, verification_code, 10)  # 10 minutes expiry
            
            # Send email
            email_sent = await email_service.send_verification_email(email, verification_code)
            
            if email_sent:
                return {
                    "success": True,
                    "message": "Verification email sent successfully",
                    "data": {"email": email}
                }
            else:
                return {"success": False, "message": "Failed to send verification email"}
                
        except Exception as e:
            return {"success": False, "message": f"Error sending verification email: {str(e)}"}

    async def verify_email_code(self, email: str, verification_code: str) -> Dict[str, Any]:
        """Verify email verification code"""
        try:
            # Get stored verification code
            stored_code = cache_utils.get_verification_code(email)
            
            if not stored_code:
                return {"success": False, "message": "Verification code not found or expired"}
            
            if stored_code != verification_code:
                return {"success": False, "message": "Invalid verification code"}
            
            # Code is valid, delete it from cache
            cache_utils.delete_verification_code(email)
            
            return {"success": True, "message": "Email verified successfully"}
            
        except Exception as e:
            return {"success": False, "message": f"Error verifying email: {str(e)}"}

    # ===== AUTHENTICATION OPERATIONS =====

    async def authenticate_user(self, phone_number: str, phone_country_code: str, otp: str = None, firebase_verified: bool = False) -> Dict[str, Any]:
        """Authenticate user with OTP only or Firebase verification"""
        try:
            user = await self.get_user_by_credentials(phone_number, phone_country_code)
            if not user:
                return {"success": False, "message": "User not found"}
            
            if not user.get("is_active"):
                return {"success": False, "message": "Account is deactivated"}
            
            # If Firebase verified, skip OTP check
            if not firebase_verified:
                # Authentication with OTP (legacy SMS-based)
                otp_verification = await self.verify_otp(phone_number, phone_country_code, otp)
                if not otp_verification["success"]:
                    return otp_verification
            else:
                # Firebase has already verified the phone number on client side
                # We just trust that Firebase did the verification
                cache_utils.delete_otp(phone_number)  # Clean up if exists
            
            # Generate tokens
            tokens = jwt_utils.create_tokens(
                str(user["_id"]),
                user["phone_number"],
                user["role"]
            )
            
            # Update login timestamp
            await self.update_login_timestamp(str(user["_id"]))
            
            return {
                "success": True,
                "message": "Authentication successful",
                "data": {
                    "user_id": str(user["_id"]),
                    "access_token": tokens["access_token"],
                    "refresh_token": tokens["refresh_token"],
                    "user": user
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"Authentication failed: {str(e)}"}

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        try:
            # Verify refresh token
            payload = jwt_utils.verify_token(refresh_token, "refresh")
            if not payload:
                return {"success": False, "message": "Invalid refresh token"}
            
            user_id = payload.get("user_id")
            user = await self.get_user_by_id(user_id)
            if not user:
                return {"success": False, "message": "User not found"}
            
            # Generate new access token
            new_access_token = jwt_utils.create_access_token({
                "user_id": user_id,
                "phone_number": user["phone_number"],
                "role": user["role"]
            })
            
            return {
                "success": True,
                "message": "Token refreshed successfully",
                "data": {
                    "access_token": new_access_token,
                    "user_id": user_id
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"Token refresh failed: {str(e)}"}

    # ===== HELPER METHODS =====

    async def get_user_with_student(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user with student profile"""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            student = None
            if user["role"] == UserRole.STUDENT.value:
                student = await self.get_student_profile(user_id)
            
            return {
                "user": user,
                "student": student
            }
            
        except Exception as e:
            return None

    async def get_users_by_role(self, role: UserRole, page: int = 1, limit: int = 50) -> List[Dict[str, Any]]:
        """Get users by role with pagination"""
        try:
            skip = (page - 1) * limit
            return get_users_by_role(self.db, role.value, limit, skip)
        except Exception as e:
            return []

    async def get_students_by_filters(self, current_class: str = None, board: str = None, 
                                    exam_target: str = None, page: int = 1, limit: int = 50) -> List[Dict[str, Any]]:
        """Get students by filters with pagination"""
        try:
            skip = (page - 1) * limit
            
            if current_class:
                return get_students_by_class(self.db, current_class, limit, skip)
            elif board:
                return get_students_by_board(self.db, board, limit, skip)
            elif exam_target:
                return get_students_by_exam_target(self.db, exam_target, limit, skip)
            else:
                return []
                
        except Exception as e:
            return []
