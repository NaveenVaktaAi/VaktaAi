# User Signup Collection Schema

## Overview
This document outlines the MongoDB collection structure for user signup and management in the VaktaAI system. The system uses two main collections: `users` (for authentication and core profile) and `students` (for educational profile details).

## Collection Structure

### 1. USERS Collection
**Collection Name:** `users`
**Purpose:** Core authentication and user profile management

#### Required Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `_id` | ObjectId | Primary key | Auto-generated |
| `phone_number` | String | 10-digit phone number | Unique, Required |
| `phone_country_code` | String | Country code | Default: "+91", Required |
| `full_name` | String | User's full name | Required |
| `password_hash` | String | Hashed password | Nullable (for OTP login) |
| `is_phone_verified` | Boolean | Phone verification status | Default: false, Required |
| `role` | String | User role | Values: student/parent/tutor/admin, Required |
| `account_status` | String | Account status | Values: trial/active/suspended/expired, Required |
| `is_active` | Boolean | Account active status | Default: true, Required |
| `created_at` | DateTime | Account creation timestamp | Auto-generated, Required |
| `last_login_at` | DateTime | Last login timestamp | Nullable |

#### Optional but Recommended Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `email` | String | User's email address | Unique, Nullable |
| `is_email_verified` | Boolean | Email verification status | Default: false |

#### Nice to Have Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `date_of_birth` | Date | User's date of birth | Nullable |
| `profile_picture_url` | String | Profile picture URL | Nullable |
| `referral_code` | String | User's referral code | Unique, Nullable |
| `referred_by` | ObjectId | Reference to referring user | Nullable |
| `device_id` | String | Device identifier | Nullable |
| `fcm_token` | String | Firebase Cloud Messaging token | Nullable |
| `last_active_at` | DateTime | Last activity timestamp | Nullable |

### 2. STUDENTS Collection
**Collection Name:** `students`
**Purpose:** Educational profile and academic information

#### Required Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `_id` | ObjectId | Primary key | Auto-generated |
| `user_id` | ObjectId | Reference to users collection | Unique, Required |
| `current_class` | String | Current academic class | Values: 6/7/8/9/10/11/12/12+, Required |
| `board` | String | Education board | Values: cbse/icse/state_board, Required |
| `exam_target` | String | Target examination | Values: boards/jee/neet/foundation/olympiad, Required |
| `preferred_language` | String | Preferred language | Values: hindi/english, Required |
| `created_at` | DateTime | Profile creation timestamp | Auto-generated, Required |
| `updated_at` | DateTime | Last update timestamp | Auto-updated |

#### Optional Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `state` | String | User's state | Nullable |
| `city` | String | User's city | Nullable |

## Database Indexes

### Users Collection Indexes
- `phone_number` (unique)
- `email` (unique, sparse)
- `role`
- `account_status`
- `is_active`
- `is_phone_verified`
- `is_email_verified`
- `created_at`
- `last_login_at`
- `referral_code` (unique, sparse)
- `referred_by`

### Students Collection Indexes
- `user_id` (unique)
- `current_class`
- `board`
- `exam_target`
- `preferred_language`
- `state`
- `city`
- `created_at`

## CRUD Operations Available

### Users Collection Operations
- `create_user(db, user_doc)` - Create new user
- `get_user(db, user_id)` - Get user by ID
- `get_user_by_phone(db, phone_number, phone_country_code)` - Get user by phone
- `get_user_by_email(db, email)` - Get user by email
- `get_users_by_role(db, role, limit, skip)` - Get users by role with pagination
- `update_user(db, user_id, fields)` - Update user
- `update_user_login(db, user_id)` - Update last login timestamp
- `update_user_active_status(db, user_id, is_active)` - Update active status
- `delete_user(db, user_id)` - Delete user and associated student profile
- `verify_phone_number(db, user_id)` - Mark phone as verified
- `verify_email(db, user_id)` - Mark email as verified

### Students Collection Operations
- `create_student(db, student_doc)` - Create new student profile
- `get_student(db, student_id)` - Get student by ID
- `get_student_by_user_id(db, user_id)` - Get student by user ID
- `get_students_by_class(db, current_class, limit, skip)` - Get students by class
- `get_students_by_board(db, board, limit, skip)` - Get students by board
- `get_students_by_exam_target(db, exam_target, limit, skip)` - Get students by exam target
- `update_student(db, student_id, fields)` - Update student profile
- `update_student_by_user_id(db, user_id, fields)` - Update student by user ID
- `delete_student(db, student_id)` - Delete student profile
- `delete_student_by_user_id(db, user_id)` - Delete student by user ID

## Usage Examples

### Creating a New User with Student Profile
```python
from datetime import datetime

# Create user document
user_doc = {
    "phone_number": "9876543210",
    "phone_country_code": "+91",
    "full_name": "John Doe",
    "password_hash": None,  # For OTP login
    "is_phone_verified": False,
    "role": "student",
    "account_status": "trial",
    "is_active": True,
    "email": "john@example.com",
    "is_email_verified": False,
    "created_at": datetime.utcnow(),
    "last_login_at": None
}

# Create user
user_id = create_user(db, user_doc)

# Create student profile
student_doc = {
    "user_id": ObjectId(user_id),
    "current_class": "10",
    "board": "cbse",
    "exam_target": "boards",
    "preferred_language": "english",
    "state": "Delhi",
    "city": "New Delhi",
    "created_at": datetime.utcnow(),
    "updated_at": datetime.utcnow()
}

student_id = create_student(db, student_doc)
```

### User Login Flow
```python
# Find user by phone
user = get_user_by_phone(db, "9876543210", "+91")

if user and user["is_active"]:
    # Update login timestamp
    update_user_login(db, str(user["_id"]))
    
    # Get student profile if role is student
    if user["role"] == "student":
        student = get_student_by_user_id(db, str(user["_id"]))
```

## Field Validation Rules

### Phone Number
- Must be exactly 10 digits
- Must be unique across all users
- Format: `[0-9]{10}`

### Email
- Must be valid email format
- Must be unique when provided
- Can be null for phone-only accounts

### Role
- Must be one of: `student`, `parent`, `tutor`, `admin`
- Case sensitive

### Account Status
- Must be one of: `trial`, `active`, `suspended`, `expired`
- Case sensitive

### Current Class
- Must be one of: `6`, `7`, `8`, `9`, `10`, `11`, `12`, `12+`
- String format

### Board
- Must be one of: `cbse`, `icse`, `state_board`
- Case sensitive

### Exam Target
- Must be one of: `boards`, `jee`, `neet`, `foundation`, `olympiad`
- Case sensitive

### Preferred Language
- Must be one of: `hindi`, `english`
- Case sensitive

## Security Considerations

1. **Password Storage**: Use bcrypt or similar for password hashing
2. **Phone Verification**: Implement OTP verification for phone numbers
3. **Email Verification**: Implement email verification links
4. **Access Control**: Implement role-based access control
5. **Data Privacy**: Ensure PII data is properly protected
6. **Audit Trail**: Log all user actions and changes

## Migration Notes

When implementing this schema:
1. Create indexes before inserting data
2. Set up proper validation rules
3. Implement proper error handling for unique constraints
4. Set up monitoring for collection performance
5. Consider data archival for inactive users
