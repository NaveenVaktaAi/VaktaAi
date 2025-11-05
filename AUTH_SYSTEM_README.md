# VaktaAI Authentication System

## Overview
This document provides comprehensive documentation for the VaktaAI authentication system, including API endpoints, database schema, and implementation details.

## System Architecture

### Components
1. **Router** (`router.py`) - FastAPI endpoints for authentication
2. **Repository** (`repository.py`) - Database operations and business logic
3. **Schema** (`schema.py`) - Pydantic models for request/response validation
4. **Utils** (`utils.py`) - Utility functions for security, JWT, OTP, email, etc.

### Database Collections
- **users** - Core user authentication and profile data
- **students** - Educational profile data for student users

## API Endpoints

### Authentication Endpoints

#### 1. User Signup
```
POST /auth/signup
```
**Request Body:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91",
  "full_name": "John Doe",
  "password": "password123",
  "role": "student",
  "email": "john@example.com",
  "current_class": "10",
  "board": "cbse",
  "exam_target": "boards",
  "preferred_language": "english",
  "state": "Delhi",
  "city": "New Delhi",
  "referral_code": "ABC12345"
}
```

**Response:**
```json
{
  "success": true,
  "message": "User created successfully",
  "data": {
    "user_id": "64f8a1b2c3d4e5f6789012ab",
    "student_id": "64f8a1b2c3d4e5f6789012cd"
  },
  "user": {
    "_id": "64f8a1b2c3d4e5f6789012ab",
    "phone_number": "9876543210",
    "phone_country_code": "+91",
    "full_name": "John Doe",
    "role": "student",
    "account_status": "trial",
    "is_active": true,
    "is_phone_verified": false,
    "email": "john@example.com",
    "is_email_verified": false,
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

#### 2. User Login
```
POST /auth/login
```
**Request Body:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91",
  "password": "password123"
}
```
or
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91",
  "otp": "123456"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "user_id": "64f8a1b2c3d4e5f6789012ab",
    "expires_in": 86400
  },
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "_id": "64f8a1b2c3d4e5f6789012ab",
    "phone_number": "9876543210",
    "full_name": "John Doe",
    "role": "student",
    "account_status": "trial",
    "is_active": true
  }
}
```

#### 3. Refresh Token
```
POST /auth/refresh-token
```
**Headers:**
```
refresh-token: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

**Response:**
```json
{
  "success": true,
  "message": "Token refreshed successfully",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "user_id": "64f8a1b2c3d4e5f6789012ab"
  }
}
```

### OTP Endpoints

#### 4. Send OTP
```
POST /auth/send-otp
```
**Request Body:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91"
}
```

#### 5. Verify OTP
```
POST /auth/verify-otp
```
**Request Body:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91",
  "otp": "123456"
}
```

#### 6. Resend OTP
```
POST /auth/resend-otp
```
**Request Body:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91"
}
```

### Password Management Endpoints

#### 7. Change Password
```
POST /auth/change-password
```
**Headers:**
```
Authorization: Bearer <access_token>
```
**Request Body:**
```json
{
  "current_password": "oldpassword123",
  "new_password": "newpassword123",
  "confirm_password": "newpassword123"
}
```

#### 8. Forgot Password
```
POST /auth/forgot-password
```
**Request Body:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91"
}
```

#### 9. Reset Password
```
POST /auth/reset-password
```
**Request Body:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91",
  "otp": "123456",
  "new_password": "newpassword123",
  "confirm_password": "newpassword123"
}
```

### Email Verification Endpoints

#### 10. Send Email Verification
```
POST /auth/send-email-verification
```
**Request Body:**
```json
{
  "email": "john@example.com"
}
```

#### 11. Verify Email
```
POST /auth/verify-email
```
**Request Body:**
```json
{
  "email": "john@example.com",
  "verification_code": "ABC123"
}
```

### User Profile Endpoints

#### 12. Get Current User Profile
```
GET /auth/me
```
**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "user": {
    "_id": "64f8a1b2c3d4e5f6789012ab",
    "phone_number": "9876543210",
    "full_name": "John Doe",
    "role": "student",
    "account_status": "trial",
    "is_active": true,
    "is_phone_verified": true,
    "email": "john@example.com",
    "is_email_verified": true,
    "created_at": "2024-01-15T10:30:00Z",
    "last_login_at": "2024-01-15T12:00:00Z"
  },
  "student": {
    "_id": "64f8a1b2c3d4e5f6789012cd",
    "user_id": "64f8a1b2c3d4e5f6789012ab",
    "current_class": "10",
    "board": "cbse",
    "exam_target": "boards",
    "preferred_language": "english",
    "state": "Delhi",
    "city": "New Delhi",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

#### 13. Update User Profile
```
PUT /auth/me
```
**Headers:**
```
Authorization: Bearer <access_token>
```
**Request Body:**
```json
{
  "full_name": "John Smith",
  "email": "johnsmith@example.com",
  "profile_picture_url": "https://example.com/profile.jpg",
  "date_of_birth": "2005-01-15",
  "state": "Maharashtra",
  "city": "Mumbai"
}
```

#### 14. Update Student Profile
```
PUT /auth/student-profile
```
**Headers:**
```
Authorization: Bearer <access_token>
```
**Request Body:**
```json
{
  "current_class": "11",
  "board": "icse",
  "exam_target": "jee",
  "preferred_language": "hindi",
  "state": "Maharashtra",
  "city": "Mumbai"
}
```

### Verification Endpoints

#### 15. Verify Phone Number
```
POST /auth/verify-phone
```
**Headers:**
```
Authorization: Bearer <access_token>
```
**Request Body:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91"
}
```

#### 16. Verify Email Address
```
POST /auth/verify-email-address
```
**Headers:**
```
Authorization: Bearer <access_token>
```
**Request Body:**
```json
{
  "email": "john@example.com"
}
```

#### 17. Logout
```
POST /auth/logout
```
**Headers:**
```
Authorization: Bearer <access_token>
```

## Database Schema

### Users Collection
```javascript
{
  "_id": ObjectId,
  "phone_number": String,           // 10 digits, unique
  "phone_country_code": String,     // Default: "+91"
  "full_name": String,              // Required
  "password_hash": String,          // Nullable for OTP login
  "is_phone_verified": Boolean,     // Default: false
  "role": String,                   // student/parent/tutor/admin
  "account_status": String,         // trial/active/suspended/expired
  "is_active": Boolean,             // Default: true
  "email": String,                  // Nullable, unique
  "is_email_verified": Boolean,     // Default: false
  "created_at": DateTime,
  "last_login_at": DateTime,        // Nullable
  "last_active_at": DateTime,       // Nullable
  "referral_code": String,          // Unique, nullable
  "referred_by": ObjectId,          // Nullable
  "profile_picture_url": String,    // Nullable
  "date_of_birth": String,          // Nullable
  "device_id": String,              // Nullable
  "fcm_token": String,              // Nullable
  "state": String,                  // Nullable
  "city": String                    // Nullable
}
```

### Students Collection
```javascript
{
  "_id": ObjectId,
  "user_id": ObjectId,              // Reference to users, unique
  "current_class": String,          // 6/7/8/9/10/11/12/12+
  "board": String,                  // cbse/icse/state_board
  "exam_target": String,            // boards/jee/neet/foundation/olympiad
  "preferred_language": String,     // hindi/english
  "state": String,                  // Nullable
  "city": String,                   // Nullable
  "created_at": DateTime,
  "updated_at": DateTime
}
```

## Security Features

### Password Security
- Bcrypt hashing with salt
- Password strength validation
- Minimum 6 characters with digits and letters

### JWT Tokens
- Access tokens (24 hours expiry)
- Refresh tokens (30 days expiry)
- Secure token generation and verification

### OTP Security
- 6-digit numeric OTPs
- 5-minute expiry
- Rate limiting (implement in production)

### Email Verification
- AWS SES integration
- 6-character alphanumeric codes
- 10-minute expiry

## Environment Variables

```bash
# Authentication
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
REFRESH_TOKEN_EXPIRE_DAYS=30

# OTP
OTP_LENGTH=6
OTP_EXPIRE_MINUTES=5

# Email
SENDER_EMAIL=noreply@vaktaai.com
EMAIL_VERIFICATION_EXPIRE_MINUTES=10

# SMS (Future integration)
SMS_SERVICE_ENABLED=false

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=vakta_ai

# AWS SES
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
```

## Error Handling

### Common HTTP Status Codes
- `200` - Success
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (invalid credentials/token)
- `403` - Forbidden (account deactivated)
- `404` - Not Found (user not found)
- `409` - Conflict (duplicate phone/email)
- `500` - Internal Server Error

### Error Response Format
```json
{
  "detail": "Error message description"
}
```

## Integration Guide

### 1. Add to Main Application
```python
from app.features.auth import auth_router

app.include_router(auth_router)
```

### 2. Database Setup
Run the MongoDB collection setup to create indexes:
```python
from app.database.mongo_collections import get_collections
from app.database.session import get_db

db = next(get_db())
get_collections(db)  # Creates collections and indexes
```

### 3. Environment Configuration
Set up the required environment variables in your `.env` file.

### 4. AWS SES Setup
1. Configure AWS SES in your AWS account
2. Verify sender email address
3. Set up proper IAM permissions
4. Update AWS credentials in environment variables

## Production Considerations

### Security
1. Use strong JWT secret keys
2. Implement rate limiting
3. Add CORS configuration
4. Use HTTPS in production
5. Implement token blacklisting for logout

### Performance
1. Use Redis for OTP caching
2. Implement database connection pooling
3. Add proper logging and monitoring
4. Use CDN for static assets

### Monitoring
1. Set up error tracking (Sentry)
2. Monitor authentication metrics
3. Track failed login attempts
4. Monitor email delivery rates

## Testing

### Unit Tests
Create tests for:
- Password hashing/verification
- JWT token generation/verification
- OTP generation/verification
- Database operations
- Email sending

### Integration Tests
Test complete authentication flows:
- Signup → Login → Profile Update
- Forgot Password → Reset Password
- Email Verification Flow

## Future Enhancements

1. **Social Login** - Google, Facebook, Apple
2. **Two-Factor Authentication** - TOTP, SMS backup
3. **Device Management** - Track and manage devices
4. **Advanced Security** - Login attempt tracking, account lockout
5. **Admin Panel** - User management interface
6. **Analytics** - User behavior tracking
7. **Notifications** - Email/SMS notifications for security events
