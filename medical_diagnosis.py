from dotenv import load_dotenv
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
import os
import openai
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings
dimension = 1536
index = faiss.IndexFlatL2(dimension)
faiss_index = FAISS(embedding_function=embeddings.embed_query, index=index, docstore=None, index_to_docstore_id={})


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
def add_medical_cases_to_index(cases):
    for case in cases:
        text = f"Symptoms: {case['symptoms']} History: {case['history']}"
        embedding = embeddings.embed_query(text)
        faiss_index.index.add(np.array([embedding], dtype=np.float32))
        faiss_index.index_to_docstore_id[len(faiss_index.index_to_docstore_id)] = case['diagnosis']

def diagnose(symptoms, history):
    query = "Symptoms: {symptoms} History: {history}"
    query_embedding = embeddings.embed_query(query)
    distances, indices = faiss_index.index.search(np.array([query_embedding], dtype=np.float32), k=3) 
    
    results = []
    for i in range(len(distances[0])):
        if indices[0][i] != -1: 
            results.append((distances[0][i], faiss_index.index_to_docstore_id[indices[0][i]]))
    
    return results

add_medical_cases_to_index(medical_cases)

st.title("Healthcare Diagnosis Support System")
st.write("Enter your symptoms and medical history:")

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
