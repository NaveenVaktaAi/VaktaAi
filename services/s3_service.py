import boto3
import os
import uuid
from botocore.exceptions import ClientError
from typing import Optional, Tuple
import logging
from app.config.settings import settings

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME
    
    def upload_file(self, file_content: bytes, original_filename: str, 
                   content_type: str) -> Tuple[bool, Optional[str]]:
        """Upload file to S3 and return success status and URL"""
        try:
            # Generate unique filename
            file_extension = original_filename.split('.')[-1]
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            s3_key = f"documents/{unique_filename}"
            
            # Upload file
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type,
                ServerSideEncryption='AES256'
            )
            
            # Generate URL
            s3_url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            
            logger.info(f"File uploaded successfully: {s3_url}")
            return True, s3_url
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return False, None
    
    def delete_file(self, s3_url: str) -> bool:
        """Delete file from S3"""
        try:
            # Extract key from URL
            s3_key = s3_url.split(f"{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/")[1]
            
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            logger.info(f"File deleted successfully: {s3_url}")
            return True
            
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            return False
    
    def get_file_content(self, s3_url: str) -> Optional[bytes]:
        """Download file content from S3"""
        try:
            # Extract key from URL
            s3_key = s3_url.split(f"{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/")[1]
            
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            return response['Body'].read()
            
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            return None

# Global S3 service instance
s3_service = S3Service()
