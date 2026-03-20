import os
import fitz  # PyMuPDF
import docx
from typing import List, Dict, Any

def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extracts text from PDF page by page."""
    docs = []
    filename = os.path.basename(file_path)
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                docs.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1
                    }
                })
        doc.close()
    except Exception as e:
        print(f"Error parsing PDF {filename}: {e}")
    return docs

def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extracts text from DOCX document."""
    docs = []
    filename = os.path.basename(file_path)
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        if full_text:
            content = "\n".join(full_text)
            docs.append({
                "content": content,
                "metadata": {
                    "source": filename,
                    "page": 1  # No native paging concept in python-docx easily extracted like PDF
                }
            })
    except Exception as e:
        print(f"Error parsing DOCX {filename}: {e}")
    return docs

def parse_text(file_path: str) -> List[Dict[str, Any]]:
    """Extracts text from TXT or Markdown files."""
    docs = []
    filename = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if content.strip():
                docs.append({
                    "content": content,
                    "metadata": {
                        "source": filename,
                        "page": 1
                    }
                })
    except Exception as e:
        print(f"Error parsing text file {filename}: {e}")
    return docs

def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Loads a document based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".txt", ".md"]:
        return parse_text(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
