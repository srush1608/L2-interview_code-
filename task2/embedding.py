import os
from sentence_transformers import SentenceTransformer
import fitz  

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()  
    return text

def chunk_text(text, max_chunk_size=500):
    """Split the text into chunks of a specified size."""
    words = text.split()
    chunks = [" ".join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size)]
    return chunks

def generate_embeddings(documents):
    all_embeddings = []
    for doc in documents:
        chunks = chunk_text(doc)
        chunk_embeddings = model.encode(chunks)
        all_embeddings.extend(chunk_embeddings)
    return all_embeddings

def generate_query_embedding(query):
    return model.encode([query]).tolist()
