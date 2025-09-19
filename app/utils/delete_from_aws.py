import boto3
from sentry_sdk import capture_exception;

from app.aws.secretKey import session

from app.config.settings import settings



s3_client = session.client('s3', region_name=settings.AWS_REGION,  config=boto3.session.Config(signature_version="v4"),)

def delete_document_from_s3(
        docName: str
):
    try:
         if docName:
          file_key = docName
          bucket_name = settings.S3_BUCKET_NAME
        # Delete the document from S3 
          s3_client.delete_object(Bucket=bucket_name, Key = f"VaktaAi/{file_key}")
          return {"message": "Document deleted successfully", "success": True}
    except Exception as e:
        print("error", e)
        capture_exception(e)
        return {"message": "Something went wrong", "success": False}
