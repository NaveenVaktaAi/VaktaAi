# Authentication System Dependencies

## Required Python Packages

Add these packages to your `requirements.txt` file:

```
# Authentication dependencies
bcrypt==4.1.2
PyJWT==2.8.0
boto3==1.34.0
botocore==1.34.0
email-validator==2.1.0
```

## Installation Commands

```bash
pip install bcrypt==4.1.2
pip install PyJWT==2.8.0
pip install boto3==1.34.0
pip install botocore==1.34.0
pip install email-validator==2.1.0
```

Or install all at once:
```bash
pip install bcrypt==4.1.2 PyJWT==2.8.0 boto3==1.34.0 botocore==1.34.0 email-validator==2.1.0
```

## Package Descriptions

### bcrypt==4.1.2
- Used for secure password hashing
- Provides salt generation and password verification
- Essential for secure password storage

### PyJWT==2.8.0
- Used for JWT token generation and verification
- Handles access and refresh token creation
- Provides secure token validation

### boto3==1.34.0
- AWS SDK for Python
- Used for AWS SES email service integration
- Handles email sending functionality

### botocore==1.34.0
- Low-level interface to AWS services
- Required by boto3
- Handles AWS service exceptions

### email-validator==2.1.0
- Email validation library
- Used by Pydantic for email field validation
- Ensures proper email format validation

## Environment Setup

After installing the dependencies, make sure to set up the following environment variables:

```bash
# AWS SES Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# JWT Configuration
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256

# Email Configuration
SENDER_EMAIL=noreply@vaktaai.com
```

## AWS SES Setup

1. **Create AWS Account** (if not already created)
2. **Set up AWS SES**:
   - Go to AWS SES console
   - Verify sender email address
   - Move out of sandbox mode (for production)
3. **Create IAM User**:
   - Create IAM user with SES permissions
   - Generate access keys
   - Update environment variables

## Verification Steps

After installation, verify the setup:

```python
# Test imports
import bcrypt
import jwt
import boto3
from botocore.exceptions import ClientError
from email_validator import validate_email

print("All dependencies installed successfully!")
```

## Troubleshooting

### Common Issues

1. **bcrypt installation fails on Windows**:
   ```bash
   pip install --only-binary=all bcrypt
   ```

2. **PyJWT version conflicts**:
   ```bash
   pip install --upgrade PyJWT
   ```

3. **boto3 AWS credentials**:
   - Ensure AWS credentials are properly configured
   - Check AWS region settings

4. **Email validation errors**:
   - Ensure email-validator is properly installed
   - Check Pydantic version compatibility

### Development vs Production

- **Development**: Use default/example credentials
- **Production**: Use proper AWS credentials and strong JWT secrets
- **Testing**: Use mock services for email/SMS in test environment
