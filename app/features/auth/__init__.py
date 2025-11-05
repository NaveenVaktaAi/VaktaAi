"""
Authentication module for VaktaAI

This module provides comprehensive authentication functionality including:
- User signup and login
- OTP verification
- Email verification
- Password management
- JWT token handling
- User profile management

Components:
- router: FastAPI routes for authentication endpoints
- repository: Database operations and business logic
- schema: Pydantic models for request/response validation
- utils: Utility functions for password hashing, JWT, OTP, email, etc.
"""

from .router import router as auth_router
from .repository import AuthRepository
from .schema import *
from .utils import *

__all__ = [
    "auth_router",
    "AuthRepository",
    # Schemas
    "UserSignupRequest",
    "UserLoginRequest", 
    "OTPVerificationRequest",
    "ResendOTPRequest",
    "ChangePasswordRequest",
    "ForgotPasswordRequest",
    "ResetPasswordRequest",
    "EmailVerificationRequest",
    "EmailVerificationConfirmRequest",
    "UserUpdateRequest",
    "StudentUpdateRequest",
    "SignupResponse",
    "LoginResponse",
    "OTPResponse",
    "AuthResponse",
    "EmailVerificationResponse",
    "UserResponse",
    "StudentResponse",
    "UserWithStudentResponse",
    # Enums
    "UserRole",
    "AccountStatus",
    "CurrentClass",
    "Board",
    "ExamTarget",
    "PreferredLanguage",
    # Utils
    "password_utils",
    "otp_utils",
    "jwt_utils",
    "referral_utils",
    "validation_utils",
    "email_service",
    "sms_utils",
    "cache_utils"
]
