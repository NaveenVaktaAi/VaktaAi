"""
PDF Text Extraction Utility for AI Tutor
Extracts text from PDF files for AI analysis
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text from a PDF file using PyPDF2
    Falls back to pdfplumber if PyPDF2 fails
    
    Args:
        pdf_path: Path to the PDF file (can be relative or absolute)
        
    Returns:
        Extracted text as string, or None if extraction fails
    """
    try:
        # Convert to absolute path if relative
        if not pdf_path.startswith('/') and not pdf_path.startswith('\\'):
            pdf_path = str(Path('.') / pdf_path.lstrip('/'))
        
        logger.info(f"📄 Extracting text from PDF: {pdf_path}")
        
        # Try PyPDF2 first (faster)
        try:
            import PyPDF2
            
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                logger.info(f"📄 PDF has {num_pages} pages")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"[Page {page_num + 1}]\n{text}")
                
                extracted_text = "\n\n".join(text_content)
                
                if extracted_text.strip():
                    logger.info(f"✅ Successfully extracted {len(extracted_text)} characters using PyPDF2")
                    return extracted_text
                else:
                    logger.warning("⚠️ PyPDF2 returned empty text, trying pdfplumber")
                    
        except ImportError:
            logger.warning("⚠️ PyPDF2 not installed, trying pdfplumber")
        except Exception as e:
            logger.warning(f"⚠️ PyPDF2 extraction failed: {e}, trying pdfplumber")
        
        # Fallback to pdfplumber (better for complex PDFs)
        try:
            import pdfplumber
            
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                logger.info(f"📄 PDF has {num_pages} pages (pdfplumber)")
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        text_content.append(f"[Page {page_num + 1}]\n{text}")
                
                extracted_text = "\n\n".join(text_content)
                
                if extracted_text.strip():
                    logger.info(f"✅ Successfully extracted {len(extracted_text)} characters using pdfplumber")
                    return extracted_text
                else:
                    logger.error("❌ pdfplumber returned empty text")
                    return None
                    
        except ImportError:
            logger.error("❌ pdfplumber not installed. Install with: pip install pdfplumber")
            return None
        except Exception as e:
            logger.error(f"❌ pdfplumber extraction failed: {e}")
            return None
            
    except FileNotFoundError:
        logger.error(f"❌ PDF file not found: {pdf_path}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error extracting PDF: {e}")
        return None


def extract_text_from_multiple_pdfs(pdf_paths: list[str]) -> dict[str, Optional[str]]:
    """
    Extract text from multiple PDF files
    
    Args:
        pdf_paths: List of PDF file paths
        
    Returns:
        Dictionary mapping PDF paths to extracted text
    """
    results = {}
    
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        results[pdf_path] = text
    
    return results


def summarize_pdf_content(pdf_text: str, max_length: int = 500) -> str:
    """
    Create a brief summary of PDF content for logging
    
    Args:
        pdf_text: Full text from PDF
        max_length: Maximum length of summary
        
    Returns:
        Truncated summary
    """
    if not pdf_text:
        return "[Empty PDF]"
    
    cleaned = " ".join(pdf_text.split())  # Remove extra whitespace
    
    if len(cleaned) <= max_length:
        return cleaned
    
    return cleaned[:max_length] + "..."

