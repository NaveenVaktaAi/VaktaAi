import hashlib
import secrets
import string
import random
import bcrypt
import jwt
import boto3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError
import logging
from pymongo.database import Database
from app.database.mongo_collections import get_user_by_phone, get_user_by_email
from app.aws.secretKey import get_secret_keys

logger = logging.getLogger(__name__)

# Get AWS credentials
keys = get_secret_keys()

# AWS SES Configuration
SES_REGION = keys.get("AWS_REGION", "us-east-1")
SES_ACCESS_KEY = keys.get("AWS_ACCESS_KEY_ID")
SES_SECRET_KEY = keys.get("AWS_SECRET_ACCESS_KEY")
SENDER_EMAIL = keys.get("SENDER_EMAIL", "connect@vaktaai.com")

# JWT Configuration
JWT_SECRET_KEY = keys.get("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30  # 30 days

# OTP Configuration
OTP_LENGTH = 6
OTP_EXPIRE_MINUTES = 5


class PasswordUtils:
    """Utility class for password operations"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise Exception("Failed to hash password")
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False


class OTPUtils:
    """Utility class for OTP operations"""
    
    @staticmethod
    def generate_otp(length: int = OTP_LENGTH) -> str:
        """Generate a random OTP"""
        return ''.join(random.choices(string.digits, k=length))
    
    @staticmethod
    def generate_verification_code(length: int = 6) -> str:
        """Generate a random verification code"""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    
    @staticmethod
    def is_otp_expired(created_at: datetime, expire_minutes: int = OTP_EXPIRE_MINUTES) -> bool:
        """Check if OTP has expired"""
        expire_time = created_at + timedelta(minutes=expire_minutes)
        return datetime.utcnow() > expire_time
    
    @staticmethod
    def get_otp_expire_time(minutes: int = OTP_EXPIRE_MINUTES) -> datetime:
        """Get OTP expiry time"""
        return datetime.utcnow() + timedelta(minutes=minutes)


class JWTUtils:
    """Utility class for JWT operations"""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            if payload.get("type") != token_type:
                return None
            return payload
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.error("Invalid token")
            return None
    
    @staticmethod
    def create_tokens(user_id: str, phone_number: str, role: str) -> Dict[str, str]:
        """Create both access and refresh tokens"""
        token_data = {
            "user_id": user_id,
            "phone_number": phone_number,
            "role": role
        }
        
        access_token = JWTUtils.create_access_token(token_data)
        refresh_token = JWTUtils.create_refresh_token(token_data)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token
        }


class ReferralUtils:
    """Utility class for referral operations"""
    
    @staticmethod
    def generate_referral_code(length: int = 8) -> str:
        """Generate a unique referral code"""
        characters = string.ascii_uppercase + string.digits
        return ''.join(random.choices(characters, k=length))
    
    @staticmethod
    def validate_referral_code(code: str) -> bool:
        """Validate referral code format"""
        if not code or len(code) != 8:
            return False
        return all(c.isalnum() and c.isupper() for c in code)


class ValidationUtils:
    """Utility class for validation operations"""
    
    @staticmethod
    def validate_phone_number(phone_number: str, country_code: str = "+91") -> bool:
        """Validate phone number format"""
        if country_code == "+91":
            return len(phone_number) == 10 and phone_number.isdigit()
        return True  # Add other country validations as needed
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        result = {
            "is_valid": True,
            "errors": []
        }
        
        if len(password) < 6:
            result["errors"].append("Password must be at least 6 characters long")
            result["is_valid"] = False
        
        if not any(c.isdigit() for c in password):
            result["errors"].append("Password must contain at least one digit")
            result["is_valid"] = False
        
        if not any(c.isalpha() for c in password):
            result["errors"].append("Password must contain at least one letter")
            result["is_valid"] = False
        
        return result
    
    @staticmethod
    def check_user_exists(db: Database, phone_number: str, email: Optional[str] = None) -> Dict[str, Any]:
        """Check if user exists by phone or email"""
        result = {
            "exists": False,
            "by_phone": False,
            "by_email": False,
            "user": None
        }
        
        # Check by phone
        user_by_phone = get_user_by_phone(db, phone_number)
        if user_by_phone:
            result["exists"] = True
            result["by_phone"] = True
            result["user"] = user_by_phone
            return result
        
        # Check by email if provided
        if email:
            user_by_email = get_user_by_email(db, email)
            if user_by_email:
                result["exists"] = True
                result["by_email"] = True
                result["user"] = user_by_email
        
        return result


class EmailService:
    """Service class for email operations using AWS SES"""
    
    def __init__(self):
        self.ses_client = boto3.client(
            'ses',
            aws_access_key_id=SES_ACCESS_KEY,
            aws_secret_access_key=SES_SECRET_KEY,
            region_name=SES_REGION
        )
    
    async def send_verification_email(self, email: str, verification_code: str) -> bool:
        """Send email verification code"""
        try:
            subject = "VaktaAI - Email Verification"
            body_html = f"""
            <html>
            <body>
                <h2>Email Verification</h2>
                <p>Thank you for signing up with VaktaAI!</p>
                <p>Your email verification code is: <strong>{verification_code}</strong></p>
                <p>This code will expire in 10 minutes.</p>
                <p>If you didn't request this, please ignore this email.</p>
                <br>
                <p>Best regards,<br>VaktaAI Team</p>
            </body>
            </html>
            """
            
            response = self.ses_client.send_email(
                Source=SENDER_EMAIL,
                Destination={'ToAddresses': [email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {'Html': {'Data': body_html, 'Charset': 'UTF-8'}}
                }
            )
            
            logger.info(f"Email sent successfully to {email}: {response['MessageId']}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to send email to {email}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email to {email}: {e}")
            return False
    
    async def send_password_reset_email(self, email: str, reset_code: str) -> bool:
        """Send password reset code"""
        try:
            subject = "VaktaAI - Password Reset"
            body_html = f"""
            <html>
            <body>
                <h2>Password Reset Request</h2>
                <p>You requested to reset your password for VaktaAI.</p>
                <p>Your password reset code is: <strong>{reset_code}</strong></p>
                <p>This code will expire in 10 minutes.</p>
                <p>If you didn't request this, please ignore this email and your password will remain unchanged.</p>
                <br>
                <p>Best regards,<br>VaktaAI Team</p>
            </body>
            </html>
            """
            
            response = self.ses_client.send_email(
                Source=SENDER_EMAIL,
                Destination={'ToAddresses': [email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {'Html': {'Data': body_html, 'Charset': 'UTF-8'}}
                }
            )
            
            logger.info(f"Password reset email sent successfully to {email}: {response['MessageId']}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to send password reset email to {email}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending password reset email to {email}: {e}")
            return False
    
    async def send_welcome_email(self, email: str, full_name: str) -> bool:
        """Send welcome email"""
        try:
            subject = "Welcome to VaktaAI!"
            body_html = f"""
            <html>
            <body>
                <h2>Welcome to VaktaAI, {full_name}!</h2>
                <p>Thank you for joining VaktaAI. We're excited to have you on board!</p>
                <p>You can now access all our features and start your learning journey.</p>
                <p>If you have any questions, feel free to contact our support team.</p>
                <br>
                <p>Best regards,<br>VaktaAI Team</p>
            </body>
            </html>
            """
            
            response = self.ses_client.send_email(
                Source=SENDER_EMAIL,
                Destination={'ToAddresses': [email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {'Html': {'Data': body_html, 'Charset': 'UTF-8'}}
                }
            )
            
            logger.info(f"Welcome email sent successfully to {email}: {response['MessageId']}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'MessageRejected':
                logger.warning(f"Email address not verified in AWS SES. Please verify {SENDER_EMAIL} and {email} in AWS SES console. Error: {e}")
            else:
                logger.error(f"Failed to send welcome email to {email}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending welcome email to {email}: {e}")
            return False


class SMSUtils:
    """Utility class for SMS operations (placeholder for SMS service integration)"""
    
    @staticmethod
    async def send_otp_sms(phone_number: str, country_code: str, otp: str) -> bool:
        """Send OTP via SMS (placeholder implementation)"""
        try:
            # TODO: Integrate with actual SMS service (Twilio, AWS SNS, etc.)
            logger.info(f"OTP {otp} sent to {country_code}{phone_number}")
            
            # For development/testing, just log the OTP
            print(f"[DEV] OTP for {country_code}{phone_number}: {otp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send OTP SMS to {country_code}{phone_number}: {e}")
            return False


class CacheUtils:
    """Utility class for cache operations (OTP storage, etc.)"""
    
    # In-memory storage for development (use Redis in production)
    _otp_storage = {}  # {phone_number: {"otp": "123456", "expires_at": datetime}}
    _verification_storage = {}  # {email: {"code": "ABC123", "expires_at": datetime}}
    
    @staticmethod
    def store_otp(phone_number: str, otp: str, expire_minutes: int = OTP_EXPIRE_MINUTES) -> None:
        """Store OTP in cache"""
        try:
            from datetime import datetime, timedelta
            expires_at = datetime.utcnow() + timedelta(minutes=expire_minutes)
            CacheUtils._otp_storage[phone_number] = {
                "otp": otp,
                "expires_at": expires_at
            }
            logger.info(f"OTP stored for {phone_number}, expires at {expires_at}")
        except Exception as e:
            logger.error(f"Error storing OTP for {phone_number}: {e}")
    
    @staticmethod
    def get_otp(phone_number: str) -> Optional[str]:
        """Get OTP from cache"""
        try:
            from datetime import datetime
            
            if phone_number not in CacheUtils._otp_storage:
                return None
            
            otp_data = CacheUtils._otp_storage[phone_number]
            
            # Check if expired
            if datetime.utcnow() > otp_data["expires_at"]:
                del CacheUtils._otp_storage[phone_number]
                logger.info(f"OTP expired for {phone_number}")
                return None
            
            return otp_data["otp"]
        except Exception as e:
            logger.error(f"Error getting OTP for {phone_number}: {e}")
            return None
    
    @staticmethod
    def delete_otp(phone_number: str) -> None:
        """Delete OTP from cache"""
        try:
            if phone_number in CacheUtils._otp_storage:
                del CacheUtils._otp_storage[phone_number]
                logger.info(f"OTP deleted for {phone_number}")
        except Exception as e:
            logger.error(f"Error deleting OTP for {phone_number}: {e}")
    
    @staticmethod
    def store_verification_code(email: str, code: str, expire_minutes: int = 10) -> None:
        """Store email verification code"""
        try:
            from datetime import datetime, timedelta
            expires_at = datetime.utcnow() + timedelta(minutes=expire_minutes)
            CacheUtils._verification_storage[email] = {
                "code": code,
                "expires_at": expires_at
            }
            logger.info(f"Verification code stored for {email}, expires at {expires_at}")
        except Exception as e:
            logger.error(f"Error storing verification code for {email}: {e}")
    
    @staticmethod
    def get_verification_code(email: str) -> Optional[str]:
        """Get email verification code"""
        try:
            from datetime import datetime
            
            if email not in CacheUtils._verification_storage:
                return None
            
            code_data = CacheUtils._verification_storage[email]
            
            # Check if expired
            if datetime.utcnow() > code_data["expires_at"]:
                del CacheUtils._verification_storage[email]
                logger.info(f"Verification code expired for {email}")
                return None
            
            return code_data["code"]
        except Exception as e:
            logger.error(f"Error getting verification code for {email}: {e}")
            return None
    
    @staticmethod
    def delete_verification_code(email: str) -> None:
        """Delete email verification code"""
        try:
            if email in CacheUtils._verification_storage:
                del CacheUtils._verification_storage[email]
                logger.info(f"Verification code deleted for {email}")
        except Exception as e:
            logger.error(f"Error deleting verification code for {email}: {e}")


# Global instances
email_service = EmailService()
password_utils = PasswordUtils()
otp_utils = OTPUtils()
jwt_utils = JWTUtils()
referral_utils = ReferralUtils()
validation_utils = ValidationUtils()
sms_utils = SMSUtils()
cache_utils = CacheUtils()
