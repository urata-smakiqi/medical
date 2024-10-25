import os
from dotenv import load_dotenv
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize FAISS index
# Create a FAISS index
dimension = 1536  # For OpenAI embeddings, typically 1536
index = faiss.IndexFlatL2(dimension)  # Using L2 distance
faiss_index = FAISS(embedding_function=embeddings.embed_query, index=index, docstore=None, index_to_docstore_id={})

# Sample medical cases (you can expand this)
medical_cases = [
    {
        "symptoms": "fever, cough, fatigue",
        "history": "Patient recently traveled to an area with a flu outbreak.",
        "diagnosis": "Flu"
    },
    {
        "symptoms": "chest pain, shortness of breath",
        "history": "Patient has a history of smoking and high cholesterol.",
        "diagnosis": "Possible heart attack"
    },
    {
        "symptoms": "headache, nausea, sensitivity to light",
        "history": "Patient has a history of migraines.",
        "diagnosis": "Migraine"
    }
]

# Function to add medical cases to the FAISS index
def add_medical_cases_to_index(cases):
    for case in cases:
        # Create a combined string of symptoms and history for embedding
        text = f"Symptoms: {case['symptoms']} History: {case['history']}"
        embedding = embeddings.embed_query(text)
        faiss_index.index.add(np.array([embedding], dtype=np.float32))  # Ensure dtype is float32
        # Store the document ID mapping
        faiss_index.index_to_docstore_id[len(faiss_index.index_to_docstore_id)] = case['diagnosis']  # Save diagnosis as ID

# Function to query the FAISS index for diagnosis
def diagnose(symptoms, history):
    query = f"Symptoms: {symptoms} History: {history}"
    query_embedding = embeddings.embed_query(query)
    distances, indices = faiss_index.index.search(np.array([query_embedding], dtype=np.float32), k=3)  # Get 3 closest results
    
    results = []
    for i in range(len(distances[0])):  # Iterate over the number of results returned
        if indices[0][i] != -1:  # Check if the index is valid
            results.append((distances[0][i], faiss_index.index_to_docstore_id[indices[0][i]]))
    
    return results

# Add medical cases to the FAISS index
add_medical_cases_to_index(medical_cases)

# Streamlit UI
st.title("Healthcare Diagnosis Support System")
st.write("Enter your symptoms and medical history:")

# Input fields for symptoms and medical history
symptoms_input = st.text_input("Symptoms")
history_input = st.text_input("Medical History")

if st.button("Get Diagnosis"):
    if symptoms_input and history_input:
        results = diagnose(symptoms_input, history_input)
        st.write("Possible Diagnoses:")
        if results:
            for _, diagnosis in results:
                st.write(f"- {diagnosis}")  # Display the diagnosis
        else:
            st.write("No diagnoses found. Please try different symptoms or history.")
    else:
        st.write("Please enter both symptoms and medical history.")
