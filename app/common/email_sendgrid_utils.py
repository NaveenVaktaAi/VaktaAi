import sendgrid
from sendgrid.helpers.mail import Content, Email, Mail, To
from sentry_sdk import capture_exception
from app.features.aws.secretKey import get_secret_keys
keys = get_secret_keys()
from app.config import env_variables


env_data = env_variables()

sg = sendgrid.SendGridAPIClient(api_key=keys.get("SENGRID_API_KEY"))


def send_email(from_email, to_email, subject, content):
    try:
        from_email = Email(keys.get("SENDER_EMAIL"))
        to_email = To(to_email)
        subject = subject
        content = Content("text/html", content)
        mail = Mail(from_email, to_email, subject, content)
        mail_json = mail.get()
        response = sg.client.mail.send.post(request_body=mail_json)

        return response.status_code
    except Exception as e:
        print(e, "Exception")
        capture_exception(e)