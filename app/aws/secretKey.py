
import json
from typing import Any, Dict

import boto3
from botocore.exceptions import NoCredentialsError
from sentry_sdk import capture_exception
import os

# from app.config import env_variables

# env_vars = env_variables()


session = boto3.Session(profile_name=os.getenv("PROFILE_NAME","default"))

secrets_client = session.client(
    "secretsmanager", region_name=os.getenv("AWS_REGION_NAME")
)

secretKeys = None


def get_secret_keys() -> Dict[Any, Any]:
    global secretKeys
    print(os.getenv("SECRET_NAME"), "==============SECRETS==============")
    if secretKeys is None:
        try:
            response = secrets_client.get_secret_value(
                SecretId=os.getenv("SECRET_NAME")
            )
            json_data = response["SecretString"]
            secretKeys = json.loads(json_data)
            print(secretKeys, "==============secretKeys==============")
        except NoCredentialsError as e:
            print(e)
        except Exception as e:
            print(e)
            capture_exception(e)
    return secretKeys or {}
