from app.aws.secretKey import session
from app.config.settings import settings
from app.database.session import get_db
import os
import re
import threading
import boto3
import fitz
import requests
from io import BytesIO
from docx import Document 
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from jinja2 import Environment, FileSystemLoader
from app.aws.secretKey import get_secret_keys
from sentry_sdk import capture_exception
from app.features.docSathi.schema import UploadDocuments
# from app.config import env_variables
import nltk
import numpy as np
from fastapi import  HTTPException
from transformers import LEDForConditionalGeneration, LEDTokenizer
import torch
from nltk.tokenize import sent_tokenize
# env_vars = env_variables()

from bson import ObjectId
from app.database import get_db
from datetime import datetime

from fastapi import  HTTPException
from transformers import LEDForConditionalGeneration, LEDTokenizer
import torch
from nltk.tokenize import sent_tokenize
from app.database.mongo_collections import (
    create_document,
    insert_chunk,
    update_document_status,
)
from bson import ObjectId
from app.schemas.milvus.collection.milvus_collections import chunk_msmarcos_collection
from app.schemas.schemas import DocSathiAIDocumentCreate
from datetime import datetime
import asyncio

from app.utils.delete_from_aws import delete_document_from_s3
from app.utils.milvus.operations.crud import insert_chunk_to_milvus
from app.database.mongo_collections import update_document_status
from bson import ObjectId


# Creating instance meaning full summary 

model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")

tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).half()

model.config.attention_window = 512 


# Creating instance meaning full chunks 
try:
    modelForChunk = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"🚨 Error loading embedding model: {e}")
    modelForChunk = None  
    



keys = get_secret_keys()

s3_client = session.client(
    "s3",
    region_name=settings.AWS_REGION,
    config=boto3.session.Config(signature_version="v4"),
)
s3_resources = session.resource(
    "s3",
    region_name=settings.AWS_REGION,
    config=boto3.session.Config(signature_version="v4"),
)

current_directory = os.path.dirname(os.path.abspath(__file__))
relative_templates_directory = "../../templates"
templates_directory = os.path.join(current_directory, relative_templates_directory)
template_env = Environment(loader=FileSystemLoader(templates_directory))


# generate presigned url
async def generate_presigned_url(filename: str):

    try:
        bucket_name = settings.S3_BUCKET_NAME
        print("bucket_name-----------", bucket_name)
        print("filename-----------", filename)
        response = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket_name, "Key": f"vaktaAi/{filename}"},
            ExpiresIn=3600,
        )

    except Exception as e:
        print("error in generate_presigned_url", e)
        capture_exception(e)
        return {"message": "Something went wrong", "success": False}

    return {"data": response, "success": True, "message": "generate_presigned_url."}





def extract_text(file_data: bytes, document_format: str) -> str:
    """
    Extract text from file based on its format.
    :param file_data: File content in bytes
    :param document_format: Format of the document (pdf/word)
    :return: Extracted text as a string
    """
    try:
        if document_format == "pdf":
            pdf_document = fitz.open(stream=file_data)
            return "".join(
                pdf_document.load_page(page_num).get_text()
                for page_num in range(len(pdf_document))
            )
        elif document_format == "docx":
            file_stream = io.BytesIO(file_data)
            doc = Document(file_stream)
            return "\n".join(
                f"\u2022 {p.text.strip()}" if p.style.name.lower().startswith("list") else p.text.strip()
                for p in doc.paragraphs
            )
        else:
            return ""
    except Exception as e:
        print(e, f"Error extracting text for format: {document_format}")
        capture_exception(e)
        return ""

def extract_text_from_docx(url: str) -> str:
    # File download from presigned URL
    response = requests.get(url)
    response.raise_for_status()

    # Load DOCX from bytes
    file_stream = BytesIO(response.content)
    doc = Document(file_stream)

    # Extract text
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


async def read_and_train_private_file(
    data: UploadDocuments,
):
    """
    Optimized function to process documents from multiple sources:
    - FileData: Multiple files uploaded to S3
    - websiteUrls: Multiple website URLs for scraping  
    - youtubeUrls: Multiple YouTube URLs for transcript extraction
    
    User can provide only one type at a time (files OR websites OR videos)
    """
    results = []

    try:
        print("Processing data:", data)
        
        # Count how many input types are provided
        input_types = 0
        if data.FileData and len(data.FileData) > 0:
            input_types += 1
        if data.WebsiteUrls and len(data.WebsiteUrls) > 0:
            input_types += 1  
        if data.YoutubeUrls and len(data.YoutubeUrls) > 0:
            input_types += 1
            
        # Validate: Only one input type allowed at a time
        if input_types == 0:
            return {
                "message": "No input data provided",
                "success": False,
                "error": "Please provide FileData, WebsiteUrls, or YoutubeUrls" 
            }
        elif input_types > 1:
            return {
                "message": "Multiple input types not allowed",
                "success": False,
                "error": "Please provide only one type: FileData OR WebsiteUrls OR YoutubeUrls"
            }
        
        # Process based on input type
        if data.FileData and len(data.FileData) > 0:
            results = await process_file_data(data.FileData, data.documentFormat, data.type)
        elif data.WebsiteUrls and len(data.WebsiteUrls) > 0:
            results = await process_website_urls(data.WebsiteUrls, data.type)
        elif data.YoutubeUrls and len(data.YoutubeUrls) > 0:
            results = await process_youtube_urls(data.YoutubeUrls, data.type)

        return {
            "message": "Document processing completed",
            "results": results,
            "success": all(result["success"] for result in results),
        }

    except Exception as e:
        print(f"Error in read_and_train_private_file: {e}")
        capture_exception(e)
        return {
            "message": "Something went wrong",
            "success": False,
            "error": str(e),
        }


async def process_file_data(file_data_list, document_format, doc_type):
    """Process multiple file uploads from S3"""
    bucket_name = settings.S3_BUCKET_NAME
    results = []
    
    for document in file_data_list:
        print(f"Processing file: {document.fileNameTime}")
        try:
            # Get file from S3
            s3_object = (
                s3_resources.Bucket(bucket_name)
                .Object(f"vaktaAi/{document.fileNameTime}")
                .get()
            )
            file_data = s3_object["Body"].read()

            # Extract text based on document format
            text = extract_text(file_data, document_format)
            if not text.strip():
                results.append({
                    "fileName": document.fileNameTime,
                    "message": "No content extracted from file",
                    "success": False,
                })
                continue

            
            # Add document record to Mongo
            db = next(get_db())
            doc_payload = DocSathiAIDocumentCreate(
                user_id=1,  # TODO: Get actual user_id
                name=document.fileNameTime,
                url=document.signedUrl,
                status="processing",
                document_format=document_format,
                type=(doc_type or "").lower() if doc_type else None,
                summary=None,
            )
            
            new_document_id = create_document(db, doc_payload.dict(exclude_none=True))

            # Train the document
            train_document(text, new_document_id, document_format or "file", "1")
            # Start training in a separate thread
            threading.Thread(
                    target=train_document,
                    args=(text, new_document_id, doc_type, "1"),
                ).start()
            results.append({
                "fileName": document.fileNameTime,
                "documentId": new_document_id,
                "documentName": document.fileNameTime,
                "message": f"File {document.fileNameTime} processed successfully",
                "success": True,
            })

        except Exception as doc_error:
            print(f"Error processing file {document.fileNameTime}: {doc_error}")
            capture_exception(doc_error)
            results.append({
                "fileName": document.fileNameTime,
                "message": f"Error processing file: {str(doc_error)}",
                "success": False,
                "error": str(doc_error),
            })
    
    return results


async def process_website_urls(website_urls, doc_type):
    """Process multiple website URLs"""
    results = []
    
    for url in website_urls:
        print(f"Processing website URL: {url.url}")
        try:
            # Scrape website content
            scraped_data = scrape_website_content(url.url)
            
            if not scraped_data['text'].strip():
                results.append({
                    "source": "website",
                    "url": url.url,
                    "message": "No content extracted from website",
                    "success": False,
                })
                continue
            
            # Generate document name from scraped content
            doc_name = generate_document_name_from_text(scraped_data['text'], "website")
            
            # Add document record to Mongo
            db = next(get_db())
            doc_payload = DocSathiAIDocumentCreate(
                user_id=1,  # TODO: Get actual user_id
                name=doc_name,
                url=url.url,
                status="processing",
                document_format="webUrl",
                type=doc_type or "website",
            )
            
            new_document_id = create_document(db, doc_payload.dict(exclude_none=True))
            print(f"Website document created with ID: {new_document_id}")
            
            # Train the document
            threading.Thread(
                    target=train_document,
                    args=(scraped_data['text'], new_document_id, doc_type, "1"),
                ).start()
            
            results.append({
                "source": "website",
                "url": url.url,
                "documentId": new_document_id,
                "documentName": doc_name,
                "message": "Website processed successfully",
                "success": True,
            })
            
        except Exception as web_error:
            print(f"Error processing website URL {url.url}: {web_error}")
            results.append({
                "source": "website",
                "url": url.url,
                "message": f"Error processing website: {str(web_error)}",
                "success": False,
                "error": str(web_error),
            })
    
    return results


async def process_youtube_urls(youtube_urls, doc_type):
    """Process multiple YouTube URLs"""
    results = []
    
    for url in youtube_urls:
        print(f"Processing YouTube URL: {url.url}")
        try:
            # Extract transcript from YouTube
            youtube_data = extract_youtube_transcript(url.url)
            
            if not youtube_data['text'].strip():
                results.append({
                    "source": "youtube",
                    "url": url.url,
                    "message": "No transcript extracted from YouTube video",
                    "success": False,
                })
                continue
            
            # Generate document name from transcript
            doc_name = generate_document_name_from_text(youtube_data['text'], "video")
            
            # Add document record to Mongo
            db = next(get_db())
            doc_payload = DocSathiAIDocumentCreate(
                user_id=1,  # TODO: Get actual user_id
                organization_id=1,  # TODO: Get actual org_id
                name=doc_name,
                url=url.url,
                status="processing",
                document_format="videoUrl",
                type=doc_type or "video",
            )
            
            new_document_id = create_document(db, doc_payload.dict(exclude_none=True))
            print(f"YouTube document created with ID: {new_document_id}")
            
            # Train the document
            threading.Thread(
                    target=train_document,
                    args=(youtube_data['text'], new_document_id, doc_type, "1"),
                ).start()
            
            results.append({
                "source": "youtube",
                "url": url.url,
                "documentId": new_document_id,
                "documentName": doc_name,
                "message": "YouTube video processed successfully",
                "success": True,
            })
            
        except Exception as youtube_error:
            print(f"Error processing YouTube URL {url.url}: {youtube_error}")
            results.append({
                "source": "youtube",
                "url": url.url,
                "message": f"Error processing YouTube video: {str(youtube_error)}",
                "success": False,
                "error": str(youtube_error),
            })
    
    return results


# Utility functions for document processing 




def train_document(text: str, document_id: str, document_type: str, org_id: str):
    print(f"Entered train_document with type {document_type} and org_id {org_id}")

    try:
        db = next(get_db()) 
        print("--------------db--------",db)
        # chunks,summary = create_text_chunks(text.lower()) 
    
        # Generate meaningful chunks
        chunks, summary = create_meaningful_chunks(text)
        print("v>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>train_document>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",chunks)
        
        # Print Chunks
        # for i, chunk in enumerate(chunks):
        #     print(f"\n🔹 Meaningful Chunk {i+1}:\n{chunk}")
        # chunks = create_text_chunks(text.lower()) 
        print("summary>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>train_document>>>>>>>>>>",summary)
        print(f"Text chunks>>>>>>>>>>>>>>>>>>>>>>train_document>>>>>>>>>>>>>>>>>>>>>>>>: {chunks}")
        if chunks:
            for item in chunks:
                chunk_id = insert_chunk(db, {
                    "detail": "Default detail",
                    "keywords": "Default keywords",
                    "meta_summary": "Default summary",
                    "chunk": item,
                    "training_document_id": ObjectId(document_id),
                    "question_id": None,
                    "organization_id": org_id,
                    "created_ts": datetime.now(),
                    "updated_ts": datetime.now(),
                })
                print("doc------------id",document_id )
                print("chunk_id------------id",chunk_id)
                asyncio.run(insert_chunk_to_milvus(db, item, document_id, chunk_id))

        # Update document status to completed
        print(f"Updating document {document_id} status to completed")
        update_document_status(db, document_id, {"status": "completed", "summary": summary, "updated_ts": datetime.now()})

    except Exception as e:
        print(f"Error in train_document: {e}")
        asyncio.run(delete_document_by_admin(document_id, False))
        update_document_status(db, document_id, {"status": "cancelled", "updated_ts": datetime.now()})
        capture_exception(e)


# delete document by document Id
async def delete_document_by_admin(docId: str, delete_document: bool = True):
    try:
        db = next(get_db())

        # Set status to deleting
        update_document_status(db, docId, {"status": "deleting", "updated_ts": datetime.now()})

        if docId:
            thread = threading.Thread(
                target=delete_data_related_document,
                args=(docId, delete_document),
            )
            thread.start()
            return {"message": "Document deleted successfully", "success": True}
        else:
            return {"message": "Document not deleted", "success": False}

    except Exception as e:
        print("error================================================", e)
        capture_exception(e)
        return {"message": "Something went wrong", "success": False}



# Delete all data related to the document (Mongo + Milvus + S3)
def delete_data_related_document(docId: str, delete_document: bool):
    print("Enter in delete_data_related_document function")
    db = next(get_db())
    try:
        # 1) Delete chunks from Milvus by document id
        chunk_ids_query = chunk_msmarcos_collection.query(
            f"document_id == {docId}",
            output_fields=["id"]
        )
        chunk_ids = [row["id"] for row in chunk_ids_query] if chunk_ids_query else []
        if chunk_ids:
            delete_response = chunk_msmarcos_collection.delete(f"id in {chunk_ids}")
            print(delete_response, "delete_response from Milvus for chunk")

        # 2) Delete chunks in Mongo tied to this document
        try:
            oid = ObjectId(docId)
        except Exception:
            print(f"Invalid ObjectId: {docId}")
            return

        chunks_result = db["chunks"].delete_many({"training_document_id": oid})
        print(f"Deleted {chunks_result.deleted_count} chunks related to document_id {docId}")

        # 3) Optionally delete document and its object from S3
        if delete_document:
            doc = db["docSathi_ai_documents"].find_one({"_id": oid}, {"name": 1})
            doc_name = doc.get("name") if doc else None

            doc_del = db["docSathi_ai_documents"].delete_one({"_id": oid})
            print(f"Deleted document: {doc_del.deleted_count} record(s)")

            if doc_name:
                try:
                    delete_document_from_s3(doc_name)
                    print("Successfully deleted data from S3")
                except Exception as s3_err:
                    print(f"S3 delete failed: {s3_err}")

        else:
            # If not physically deleting the document, mark as deleted
            db["docSathi_ai_documents"].update_one(
                {"_id": oid},
                {"$set": {"status": "deleted", "updated_ts": datetime.now()}}
            )

        print("All data related to document deleted successfully.")

    except Exception as e:
        capture_exception(e)
        print(e, "error in delete_data_related_document")



# Function to determine adaptive chunk size based on document length
def get_dynamic_chunk_size(word_count):
    """
    Dynamically determine chunk size based on the document length.
    Ensures chunking is meaningful based on sentence structure.
    """
    if word_count <= 1000:
        return 250  # Small documents → smaller chunks
    elif word_count <= 5000:
        return 500  # Medium-sized documents
    elif word_count <= 10000:
        return 750  # Longer documents
    else:
        return 1000  # Very large documents → bigger chunks
    



def create_meaningful_chunks(text, similarity_threshold=0.5):
    """
    Splits unstructured text into meaningful chunks while ensuring topic coherence.
    Uses SentenceTransformer embeddings for similarity calculations.
    """
    try:

        print("entring meaningfull chunk function--------------")
        # ✅ Input Validation
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text must be a non-empty string.")

        # ✅ Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        print("meaningful text1----------------",text)

        # ✅ Sentence Tokenization (More Accurate)
        sentences = sent_tokenize(text)  # Extracts sentences correctly
        print("sentences-----------------",sentences)
        print(sentences, 'sentencessentences>>>>>>>>>>>>>>>')
        # ✅ Adaptive Chunk Size
        document_length = len(text.split())  # Count total words
        chunk_size = get_dynamic_chunk_size(document_length)
        print("📌 Adaptive chunk size:", chunk_size)

        # ✅ Ensure model is loaded before encoding
        if modelForChunk is None:
            raise RuntimeError("Sentence Transformer model failed to load.")

        # ✅ Encode sentences efficiently
        try:
            sentence_embeddings = modelForChunk.encode(sentences, batch_size=8)
        except Exception as e:
            raise RuntimeError(f"🚨 Error in sentence encoding: {e}")

        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence.split())

            # ✅ Handling Very Long Sentences (Ensuring Proper Splitting)
            if sentence_length > chunk_size * 1.5:
                print(f"⚠️ Warning: Long sentence detected, ensuring complete split.")
                split_sentences = sent_tokenize(sentence)  # Split by sentence, not words
                for sub_sentence in split_sentences:
                    chunks.append(sub_sentence)
                continue  # Skip to next iteration

            # ✅ Compute similarity with previous sentence (If applicable)
            if i > 0:
                try:
                    similarity = np.dot(sentence_embeddings[i], sentence_embeddings[i-1]) / (
                        np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[i-1]) + 1e-6
                    )  # Small value added to avoid division by zero
                except Exception as e:
                    print(f"⚠️ Warning: Error computing similarity, defaulting to 1.0. Error: {e}")
                    similarity = 1.0  # Assume high similarity if error occurs

                # ✅ If similarity is LOW, start a new chunk
                if similarity < similarity_threshold and current_length > chunk_size * 0.75:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

            # ✅ Add Sentence to Current Chunk
            current_chunk.append(sentence)
            current_length += sentence_length

            # ✅ If chunk size limit is reached, finalize the chunk (ENSURE FULL SENTENCE)
            if current_length >= chunk_size and "." in sentence:  # Ensure it ends at a period
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        # ✅ Add the last chunk if any remains
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # ✅ Generate summary AFTER chunking for better context
        summary = create_summary(" ".join(chunks))  # Generate summary based on final chunks

        return chunks, summary

    except Exception as e:
        print(f"🚨 Unexpected error in create_meaningful_chunks: {e}")
        return [], ""




def create_summary(text: str) -> str:
    try:
        word_count = len(text.split())  # Get document size
        min_length, max_length = get_summary_length(word_count)  # Dynamic summary size

        chunks = chunk_text(text, max_tokens=10000)  # Adjust chunk size
        summaries = []

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")  # Debugging: print chunk progress
            inputs = tokenizer(chunk, return_tensors="pt", max_length=10000,padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            

            # Generate Summary with Optimized Parameters
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=5,  # More beams for better sentence generation
                length_penalty=1.0,  # Encourages complete sentences
                no_repeat_ngram_size=3,  # Avoids repeating words
                early_stopping=False  # Prevents premature summary cutoff
            )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        return " ".join(summaries)

    except Exception as e:
        print(f"🚨 Error in create_summary: {e}")
        return "Could not generate summary due to an error."
 




 # 🔹 Optimized Function to Dynamically Adjust Summary Length
def get_summary_length(word_count):
    if word_count <= 1000:
        return 100, 200  # Short documents
    elif word_count <= 5000:
        return 150, 300  # Medium documents
    elif word_count <= 10000:
        return 300, 600  # Long documents
    else:
        return 500, 1000  # Very Large documents


# 🔹 Improved Chunking Function (Ensures Sentence Completion & Smart Token Limits)
def chunk_text(text, max_tokens=10000):
    sentences = sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(tokenized_sentence)

        # If adding this sentence exceeds max_tokens, finalize the chunk
        if current_length + sentence_length > max_tokens:
            if current_chunk:  # Ensure we don't create empty chunks
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]  # Start a new chunk with this sentence
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add last chunk if any sentences remain
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def scrape_website_content(url):
    """
    Scrape content from a website URL and extract meaningful text.
    """
    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Get text from main content areas
        content_selectors = [
            'main', 'article', '.content', '#content', '.post', '.entry-content',
            '.article-content', '.page-content', 'section'
        ]
        
        main_content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = " ".join([elem.get_text(strip=True) for elem in elements])
                break
        
        # If no main content found, get all paragraph text
        if not main_content:
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            main_content = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        # Clean up text
        main_content = re.sub(r'\s+', ' ', main_content)  # Remove extra whitespace
        main_content = main_content.strip()
        
        # Get page title for document name
        title = soup.find('title')
        page_title = title.get_text().strip() if title else urlparse(url).netloc
        
        return {
            'text': main_content,
            'title': page_title,
            'url': url
        }
        
    except Exception as e:
        print(f"Error scraping website {url}: {e}")
        return {
            'text': "",
            'title': f"Website - {urlparse(url).netloc}",
            'url': url
        }


def generate_document_name_from_text(text, source_type="document"):
    """
    Generate a meaningful document name based on text content.
    """
    try:
        # Get first few sentences or words
        sentences = sent_tokenize(text)
        if sentences:
            # Use first sentence, but limit length
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 100:
                # Take first 100 characters and find last complete word
                truncated = first_sentence[:100]
                last_space = truncated.rfind(' ')
                if last_space > 50:  # Ensure we have meaningful content
                    first_sentence = truncated[:last_space]
            
            # Clean up the name
            name = re.sub(r'[^\w\s-]', '', first_sentence)
            name = re.sub(r'\s+', ' ', name).strip()
            
            if name:
                return f"{source_type.title()}: {name}"
        
        # Fallback: use first few words
        words = text.split()[:10]
        if words:
            name = " ".join(words)
            name = re.sub(r'[^\w\s-]', '', name)
            return f"{source_type.title()}: {name}"
            
    except Exception as e:
        print(f"Error generating document name: {e}")
    
    # Ultimate fallback
    return f"{source_type.title()} - {datetime.now().strftime('%Y%m%d_%H%M%S')}"


def extract_youtube_transcript(youtube_url):
    """
    Extract transcript from YouTube URL using youtube-transcript-api.
    """
    try:
        # Import here to avoid dependency issues if not installed
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Extract video ID from URL
        video_id = extract_youtube_video_id(youtube_url)
        if not video_id:
            return {
                'text': "",
                'title': f"YouTube Video - {youtube_url}",
                'url': youtube_url
            }
        
        # Try to get transcript in different languages
        languages = ['en', 'hi', 'en-US', 'en-GB']  # English and Hindi
        transcript_text = ""
        
        for lang in languages:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                transcript_text = " ".join([entry['text'] for entry in transcript])
                if transcript_text.strip():
                    break
            except:
                continue
        
        # If no transcript found in preferred languages, try any available
        if not transcript_text.strip():
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([entry['text'] for entry in transcript])
            except:
                pass
        
        # Clean up transcript text
        transcript_text = re.sub(r'\[.*?\]', '', transcript_text)  # Remove [Music], [Applause] etc.
        transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()
        
        return {
            'text': transcript_text,
            'title': f"YouTube Video - {video_id}",
            'url': youtube_url
        }
        
    except ImportError:
        print("youtube-transcript-api not installed. Please install it: pip install youtube-transcript-api")
        return {
            'text': "",
            'title': f"YouTube Video - {youtube_url}",
            'url': youtube_url
        }
    except Exception as e:
        print(f"Error extracting YouTube transcript: {e}")
        return {
            'text': "",
            'title': f"YouTube Video - {youtube_url}",
            'url': youtube_url
        }


def extract_youtube_video_id(url):
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:v\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
