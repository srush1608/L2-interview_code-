from sentence_transformers import SentenceTransformer
import fitz  

model = SentenceTransformer('all-MiniLM-L6-v2')  

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
test_embedding = model.encode(["Test document"])
print(len(test_embedding))  # This should print 384

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()  
    return text

def generate_embeddings(documents):
    embeddings = model.encode(documents)  # Generates embeddings for each document
    return embeddings.tolist()  # Convert to list to be compatible with PostgreSQL pgvector

def generate_query_embedding(query):
    return model.encode([query]).tolist()  # Convert query to embedding and return

