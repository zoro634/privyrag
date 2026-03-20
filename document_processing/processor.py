import os
from typing import List, Dict, Any
from .parsers import load_document
from .chunker import chunk_document

def process_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Parses a single document and splits it into chunks.
    """
    pages = load_document(file_path)
    chunks = chunk_document(pages, chunk_size, chunk_overlap)
    return chunks

def process_batch(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Processes a list of files sequentially.
    """
    all_chunks = []
    for fp in file_paths:
        try:
            chunks = process_document(fp, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {fp} in batch: {e}")
    return all_chunks
