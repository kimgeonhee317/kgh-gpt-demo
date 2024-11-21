import os
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from langchain_community.embeddings import ClovaXEmbeddings

# Load the API key from env variables
load_dotenv()
chromadb.api.client.SharedSystemClient.clear_system_cache()

def create_vector_database(chunks):
    # Define embedding model
    clovax_embeddings = ClovaXEmbeddings(model='bge-m3')
    ClovaXEmbeddings.Config.protected_namespaces = ()

    # Specify local client path
    client = chromadb.PersistentClient(path="./chroma_langchain_db", settings=Settings(anonymized_telemetry=False))

    # Create Chroma collection
    chroma_collection = client.get_or_create_collection(
        name="clovastudiodatas_docs",
        metadata={"hnsw:space": "cosine"}
    )

    # Initialize Streamlit progress bar
    progress_bar = st.progress(0)
    total_documents = len(chunks)
    
    # Add documents loop with Streamlit progress bar
    for index, doc in tqdm(enumerate(chunks)):
        embeddings = clovax_embeddings.embed_documents([doc.page_content])[0]
        
        # Add document to the vector database
        chroma_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[doc.page_content],
            embeddings=[embeddings],
            metadatas=[doc.metadata]
        )
        time.sleep(1.1)  # Controlled delay for usage limits
        
        # Update progress bar
        progress_bar.progress((index + 1) / total_documents)
