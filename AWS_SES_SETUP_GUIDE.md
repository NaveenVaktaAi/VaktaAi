# AWS SES Setup Guide for VaktaAI

## Issue Resolution

The error you're seeing:
```
Email address is not verified. The following identities failed the check in region US-EAST-1: naveen.sharma@vaktaai.com, naveensharma.cse23@jecrc.ac.in
```

This means both the sender email (`naveen.sharma@vaktaai.com`) and recipient email (`naveensharma.cse23@jecrc.ac.in`) need to be verified in AWS SES.

## Quick Fix Steps

### 1. **Verify Sender Email (naveen.sharma@vaktaai.com)**

1. Go to [AWS SES Console](https://console.aws.amazon.com/ses/)
2. Make sure you're in the **US-East-1 (N. Virginia)** region
3. Click on **"Verified identities"** in the left sidebar
4. Click **"Create identity"**
5. Select **"Email address"**
6. Enter: `naveen.sharma@vaktaai.com`
7. Click **"Create identity"**
8. Check your email inbox for verification email
9. Click the verification link in the email

### 2. **Verify Recipient Email (naveensharma.cse23@jecrc.ac.in)**

1. In the same SES console
2. Click **"Create identity"** again
3. Enter: `naveensharma.cse23@jecrc.ac.in`
4. Click **"Create identity"**
5. Check the email inbox for verification email
6. Click the verification link

### 3. **Move Out of Sandbox Mode (For Production)**

AWS SES starts in sandbox mode, which means:
- You can only send emails to verified email addresses
- You're limited to 200 emails per day
- You can only send 1 email per second

To move out of sandbox mode:
1. In SES console, go to **"Account dashboard"**
2. Click **"Request production access"**
3. Fill out the form explaining your use case
4. Wait for AWS approval (usually 24-48 hours)

## Alternative: Development Mode Setup

For development/testing, you can use a different approach:

### Option 1: Use a Verified Domain
1. Verify your domain (e.g., `vaktaai.com`) instead of individual emails
2. This allows sending to any email address under that domain

### Option 2: Use Gmail SMTP (Temporary)
Update the email service to use Gmail SMTP for development:

```python
# In utils.py, add this alternative email service
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class GmailService:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email = "your-gmail@gmail.com"
        self.password = "your-app-password"  # Use app password, not regular password
    
    async def send_email(self, to_email: str, subject: str, body: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            text = msg.as_string()
            server.sendmail(self.email, to_email, text)
            server.quit()
            
            return True
        except Exception as e:
            logger.error(f"Gmail send failed: {e}")
            return False
```

## Environment Variables Update

Make sure your environment variables are set correctly:

```bash
# AWS SES Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# Email Configuration
SENDER_EMAIL=naveen.sharma@vaktaai.com  # This should be verified in SES
```

## Testing Without Email

For now, you can test the signup without email verification:

1. **Signup without email:**
```json
{
  "phone_number": "9876543210",
  "phone_country_code": "+91",
  "full_name": "Test User",
  "password": "test123456",
  "role": "student",
  "current_class": "10",
  "board": "cbse",
  "exam_target": "boards",
  "preferred_language": "english"
}
```

2. **Or signup with a verified email address**

## Current Status

The authentication system is working correctly. The email sending failure doesn't affect the core functionality:

✅ **Working:**
- User signup
- Database storage
- JWT token generation
- Profile creation
- All API endpoints

⚠️ **Email Issue:**
- Welcome emails will fail until SES is properly configured
- This is non-blocking (signup still succeeds)

## Next Steps

1. **Immediate:** Verify the email addresses in AWS SES
2. **Short-term:** Test with verified emails or disable email sending
3. **Long-term:** Move to production SES or implement domain verification

The authentication system is fully functional - the email issue is just a configuration matter that can be resolved quickly!
