import os
import psycopg2
import numpy as np
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

def get_db_connection():
    """Create and return a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def initialize_db():
    conn = get_db_connection()
    if conn is not None:
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create a new table with the correct embedding dimension
        cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                document_name TEXT,
                content TEXT,
                embedding vector(384)  
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()

def store_embeddings(documents, embeddings):
    conn = get_db_connection()
    if conn is not None:
        cursor = conn.cursor()

        for i, doc in enumerate(documents):
            document_name = f"Document_{i+1}"
            document_content = doc[:50]  # Adjust as needed
            embedding_list = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]

            cursor.execute("""
                INSERT INTO document_embeddings (document_name, content, embedding)
                VALUES (%s, %s, %s)
            """, (document_name, document_content, embedding_list))

        conn.commit()
        cursor.close()
        conn.close()

def query_database(query_embedding, top_k=1):
    """Query the database for the most relevant document based on the query embedding."""
    conn = get_db_connection()
    if conn is not None:
        cursor = conn.cursor()

        # Convert query_embedding to a flattened 1-D list
        query_embedding_list = query_embedding.flatten().tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        # Find the most relevant document(s) based on vector similarity
        cursor.execute("""
            SELECT id, document_name, content, embedding,
                   (embedding <=> %s::vector) AS distance
            FROM document_embeddings
            ORDER BY distance
            LIMIT %s;
        """, (query_embedding_list, top_k))

        results = cursor.fetchall()
        cursor.close()
        conn.close()

        if results:
            response = " | ".join(
                f"Document ID: {result[0]}, Document Name: {result[1]}, Content: {result[2]}, Similarity Distance: {result[4]}"
                for result in results
            )
            return response
        else:
            return "No relevant document found."
    else:
        return "Failed to connect to the database."

# Example setup and usage
if __name__ == "__main__":
    initialize_db()
    
    # Example documents and embeddings
    documents = ["This is a sample document on blood pressure medicine."]
    embeddings = [np.random.rand(384)]  

    store_embeddings(documents, embeddings)

    # Query with a sample embedding
    query_embedding = np.random.rand(384)  # Replace with an actual query embedding of the same dimension
    response = query_database(query_embedding)
    print(response)
