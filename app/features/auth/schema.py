from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional
from datetime import datetime
from enum import Enum
from bson import ObjectId


class UserRole(str, Enum):
    """User roles enumeration"""
    STUDENT = "student"
    PARENT = "parent"
    TUTOR = "tutor"
    ADMIN = "admin"


class AccountStatus(str, Enum):
    """Account status enumeration"""
    TRIAL = "trial"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


class CurrentClass(str, Enum):
    """Current class enumeration"""
    CLASS_6 = "6"
    CLASS_7 = "7"
    CLASS_8 = "8"
    CLASS_9 = "9"
    CLASS_10 = "10"
    CLASS_11 = "11"
    CLASS_12 = "12"
    CLASS_12_PLUS = "12+"


class Board(str, Enum):
    """Education board enumeration"""
    CBSE = "cbse"
    ICSE = "icse"
    STATE_BOARD = "state_board"


class ExamTarget(str, Enum):
    """Exam target enumeration"""
    BOARDS = "boards"
    JEE = "jee"
    NEET = "neet"
    FOUNDATION = "foundation"
    OLYMPIAD = "olympiad"


class PreferredLanguage(str, Enum):
    """Preferred language enumeration"""
    HINDI = "hindi"
    ENGLISH = "english"


# ===== REQUEST SCHEMAS =====

class UserSignupRequest(BaseModel):
    """Schema for user signup request"""
    phone_number: str = Field(..., min_length=10, max_length=10, description="10-digit phone number")
    phone_country_code: str = Field(default="+91", description="Country code")
    full_name: str = Field(..., min_length=2, max_length=100, description="User's full name")
    role: UserRole = Field(..., description="User role")
    email: Optional[EmailStr] = Field(None, description="User's email address (optional)")
    referral_code: Optional[str] = Field(None, description="Referral code")
    
    # Student-specific fields
    current_class: Optional[CurrentClass] = Field(None, description="Current academic class")
    board: Optional[Board] = Field(None, description="Education board")
    exam_target: Optional[ExamTarget] = Field(None, description="Target examination")
    preferred_language: Optional[PreferredLanguage] = Field(None, description="Preferred language")
    state: Optional[str] = Field(None, description="User's state")
    city: Optional[str] = Field(None, description="User's city")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not v.isdigit():
            raise ValueError('Phone number must contain only digits')
        return v
    
    @validator('current_class', 'board', 'exam_target', 'preferred_language')
    def validate_student_fields(cls, v, values):
        role = values.get('role')
        if role == UserRole.STUDENT and v is None:
            raise ValueError(f'Field is required when role is student')
        return v


class UserLoginRequest(BaseModel):
    """Schema for user login request"""
    phone_number: str = Field(..., min_length=10, max_length=10, description="10-digit phone number")
    phone_country_code: str = Field(default="+91", description="Country code")
    otp: str = Field(..., min_length=4, max_length=20, description="OTP for login (or firebase_verified for Firebase auth)")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not v.isdigit():
            raise ValueError('Phone number must contain only digits')
        return v


class OTPVerificationRequest(BaseModel):
    """Schema for OTP verification request"""
    phone_number: str = Field(..., min_length=10, max_length=10, description="10-digit phone number")
    phone_country_code: str = Field(default="+91", description="Country code")
    otp: str = Field(..., min_length=4, max_length=6, description="OTP code")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not v.isdigit():
            raise ValueError('Phone number must contain only digits')
        return v


class ResendOTPRequest(BaseModel):
    """Schema for resend OTP request"""
    phone_number: str = Field(..., min_length=10, max_length=10, description="10-digit phone number")
    phone_country_code: str = Field(default="+91", description="Country code")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not v.isdigit():
            raise ValueError('Phone number must contain only digits')
        return v




class EmailVerificationRequest(BaseModel):
    """Schema for email verification request"""
    email: EmailStr = Field(..., description="Email address to verify")


class EmailVerificationConfirmRequest(BaseModel):
    """Schema for email verification confirmation request"""
    email: EmailStr = Field(..., description="Email address")
    verification_code: str = Field(..., min_length=6, max_length=6, description="Verification code")


# ===== RESPONSE SCHEMAS =====

class UserResponse(BaseModel):
    """Schema for user response"""
    id: str = Field(..., alias="_id")
    phone_number: str
    phone_country_code: str
    full_name: str
    role: UserRole
    account_status: AccountStatus
    is_active: bool
    is_phone_verified: bool
    email: Optional[str] = None
    is_email_verified: Optional[bool] = None
    created_at: datetime
    last_login_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    referral_code: Optional[str] = None
    referred_by: Optional[str] = None
    profile_picture_url: Optional[str] = None
    date_of_birth: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class StudentResponse(BaseModel):
    """Schema for student profile response"""
    id: str = Field(..., alias="_id")
    user_id: str
    current_class: CurrentClass
    board: Board
    exam_target: ExamTarget
    preferred_language: PreferredLanguage
    state: Optional[str] = None
    city: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class UserWithStudentResponse(BaseModel):
    """Schema for user with student profile response"""
    user: UserResponse
    student: Optional[StudentResponse] = None


class LoginResponse(BaseModel):
    """Schema for login response"""
    success: bool
    message: str
    data: dict
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user: Optional[UserResponse] = None


class SignupResponse(BaseModel):
    """Schema for signup response"""
    success: bool
    message: str
    data: Optional[dict] = None
    user: Optional[UserResponse] = None
    verification_email_sent: bool = False


class OTPResponse(BaseModel):
    """Schema for OTP response"""
    success: bool
    message: str
    data: dict
    otp_sent: bool = False
    expires_in: int = 300  # 5 minutes in seconds


class EmailVerificationResponse(BaseModel):
    """Schema for email verification response"""
    success: bool
    message: str
    data: dict
    verification_sent: bool = False


# ===== UPDATE SCHEMAS =====

class UserUpdateRequest(BaseModel):
    """Schema for user update request"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    profile_picture_url: Optional[str] = None
    date_of_birth: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None


class StudentUpdateRequest(BaseModel):
    """Schema for student profile update request"""
    current_class: Optional[CurrentClass] = None
    board: Optional[Board] = None
    exam_target: Optional[ExamTarget] = None
    preferred_language: Optional[PreferredLanguage] = None
    state: Optional[str] = None
    city: Optional[str] = None


# ===== COMMON RESPONSE SCHEMA =====

class AuthResponse(BaseModel):
    """Common authentication response schema"""
    success: bool
    message: str
    data: Optional[dict] = None



class CurrentUser(BaseModel):
    id: int
    email: str
    role_type : str