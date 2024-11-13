from embedding import generate_embeddings, generate_query_embedding, extract_text_from_pdf
from database import initialize_db, store_embeddings, query_database
import numpy as np

# List of PDF files to process
pdf_files = [
    "Dengue.pdf",
    "High-Blood-Pressure.pdf",
    "BP2-medicine.pdf"
]

# Initialize the database and create the table
initialize_db()
print("Database initialized and table created.")


# Extract text from each PDF file
documents = []
for pdf_file in pdf_files:
    pdf_text = extract_text_from_pdf(pdf_file)
    documents.append(pdf_text)
    print(pdf_text)
print("Extracted text from PDF files.")



# Generate embeddings for the documents
embeddings = generate_embeddings(documents)
print("Generated embeddings for documents.")

# Store the documents and embeddings in the PostgreSQL database
store_embeddings(documents, embeddings)
print("Documents and embeddings stored in the database.")

# Prompt for a query
query = input("Enter your query: ")

# Generate embedding for the query
query_embedding = generate_query_embedding(query)

# Debugging: Print the shape of the query embedding
print(f"Query embedding shape: {np.array(query_embedding).shape}")  # Confirm the shape of the query embedding

# Ensure query_embedding is a NumPy array
if isinstance(query_embedding, list):
    query_embedding = np.array(query_embedding)  # Convert to NumPy array if it's a list

# Flatten the embedding if it's not 1-D
if query_embedding.ndim > 1:
    query_embedding = query_embedding.flatten()  # Flatten to 1-D

# Ensure query_embedding is a list before passing to the database
query_embedding = query_embedding.tolist()  # Convert to list

# Query the database for the most relevant document
response = query_database(query_embedding)

# Print the response
print("Query Result:")
print(response)







# @app.route('/initialize', methods=['GET'])
# def initialize():
#     """Endpoint to initialize the database and create the necessary table."""
#     initialize_db()
#     return jsonify({"status": "Database initialized and table created."})

# @app.route('/load-documents', methods=['POST'])
# def load_documents():
#     """Endpoint to load documents from PDF files, generate embeddings, and store them."""
#     documents = []
#     for pdf_file in pdf_files:
#         pdf_text = extract_text_from_pdf(pdf_file)
#         documents.append(pdf_text)

#     # Generate embeddings for the documents
#     embeddings = generate_embeddings(documents)

#     # Store the documents and embeddings in the PostgreSQL database
#     store_embeddings(documents, embeddings)
#     return jsonify({"status": "Documents loaded and embeddings stored in the database."})

# @app.route('/query', methods=['GET'])
# def query():
#     """Endpoint to perform a query based on the user input and return the most relevant document."""
#     query_text = request.args.get('query')
#     if not query_text:
#         return jsonify({"error": "Query text is required"}), 400

#     # Generate embedding for the query
#     query_embedding = generate_query_embedding(query_text)

#     # Query the database for the most relevant document
#     response = query_database(query_embedding)

#     # Return the response as JSON
#     return jsonify({"query": query_text, "response": response})

# if __name__ == "__main__":
#     app.run(debug=True)
