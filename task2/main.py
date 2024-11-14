
# Method 1 using enpoint
import os
from dotenv import load_dotenv
import numpy as np
from embedding import generate_embeddings, generate_query_embedding, extract_text_from_pdf
from database import initialize_db, store_embeddings, query_database
import requests
from flask import Flask, render_template, request

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def call_groq_api(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return "Error generating response from Groq API. Please try again later."
    
    except requests.exceptions.RequestException as e:
        print(f"RequestException: {e}")
        return "Network error. Unable to connect to Groq API. Please try again later."

# Initialize the database and process the PDFs
initialize_db()
print("Database initialized and table created.")

pdf_files = ["Dengue.pdf", "High-Blood-Pressure.pdf", "BP2-medicine.pdf"]
documents = []
for pdf_file in pdf_files:
    pdf_text = extract_text_from_pdf(pdf_file)
    documents.append(pdf_text)
print("Extracted text from PDF files.")

embeddings = generate_embeddings(documents)
print("Generated embeddings for document chunks.")

store_embeddings(documents, embeddings)
print("Document chunks and embeddings stored in the database.")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["query"]
        
        # Generate the query embedding
        query_embedding = generate_query_embedding(user_query)
        query_embedding = np.array(query_embedding).flatten().tolist()

        # Query the database with the generated embedding
        db_response = query_database(query_embedding)
        print("Database Query Result:", db_response)
        
        # Create a specific prompt for the LLM
        prompt = f"Based on the content: {db_response}, answer the following question: {user_query}"
        
        # Get the response from the Groq API
        llm_response = call_groq_api(prompt)

        return render_template("index.html", query=user_query, result=llm_response)
    
    return render_template("index.html", query="", result="")

if __name__ == "__main__":
    app.run(debug=True)

# Method 2 using print response in console
# import os
# from dotenv import load_dotenv
# import numpy as np
# from embedding import generate_embeddings, generate_query_embedding, extract_text_from_pdf
# from database import initialize_db, store_embeddings, query_database
# import requests

# load_dotenv()
# GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# def call_groq_api(prompt):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {GROQ_API_KEY}"
#     }

#     payload = {
#         "model": "llama3-8b-8192",
#         "messages": [
#             {"role": "user", "content": prompt}
#         ]
#     }

#     try:
#         url = "https://api.groq.com/openai/v1/chat/completions"
        
#         response = requests.post(url, headers=headers, json=payload)
        
#         if response.status_code == 200:
#             return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
#         else:
#             print(f"Error: {response.status_code}, {response.text}")
#             return "Error generating response from Groq API. Please try again later."
    
#     except requests.exceptions.RequestException as e:
#         print(f"RequestException: {e}")
#         return "Network error. Unable to connect to Groq API. Please try again later."

# # Initialize the database and process the PDFs
# initialize_db()
# print("Database initialized and table created.")

# pdf_files = ["Dengue.pdf", "High-Blood-Pressure.pdf", "BP2-medicine.pdf"]
# documents = []
# for pdf_file in pdf_files:
#     pdf_text = extract_text_from_pdf(pdf_file)
#     documents.append(pdf_text)
# print("Extracted text from PDF files.")

# embeddings = generate_embeddings(documents)
# print("Generated embeddings for document chunks.")

# store_embeddings(documents, embeddings)
# print("Document chunks and embeddings stored in the database.")

# # Console query method (commented out for now)
# def console_query():
#     user_query = input("Enter your query: ")

#     query_embedding = generate_query_embedding(user_query)
#     query_embedding = np.array(query_embedding).flatten().tolist()

#     db_response = query_database(query_embedding)
#     print("Database Query Result:", db_response)

#     # Modify the prompt to be specific
#     prompt = f"Based on the content: {db_response}, answer the following question: {user_query}"

#     llm_response = call_groq_api(prompt)

#     print("LLM Response from Groq API:")
#     print(llm_response)

# if __name__ == "__main__":
#     # Uncomment the following line to run the console method
#     # console_query()
#     pass
