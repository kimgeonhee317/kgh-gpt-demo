import streamlit as st
import chromadb
from chromadb.config import Settings
import os
from src.global_settings import DB_PATH
from langchain_chroma import Chroma

def get_embedded_documents():
    # Initialize Chroma PersistentClient

    db_file_path = DB_PATH 
    if not os.path.exists(db_file_path):
        st.warning("벡터 데이터베이스 파일이 존재하지 않습니다.")
        return

    client = chromadb.PersistentClient(path="./chroma_langchain_db", settings=Settings(anonymized_telemetry=False))

    # Access the existing collection
    collection_name = "clovastudiodatas_docs"
    collection = client.get_or_create_collection(name=collection_name)

    # Function to list all documents in the collection
    def list_documents():
        # Retrieve all document content and their corresponding metadata
        docs = collection.get(include=["documents", "metadatas"])
        
        documents = docs["documents"]
        metadatas = docs["metadatas"]
        
        # Combine documents and metadata for display
        meta_list = []
        doc_list = []
        for doc, meta in zip(documents, metadatas):
            if meta['source'] not in meta_list:
                meta_list.append(meta['source'])
                doc_list.append({"Content": doc, "Metadata": meta})
    
        return doc_list

    # Fetch and display the documents
    documents = list_documents()
    if documents:
        for idx, doc_info in enumerate(documents):
            st.write(f"**Document {idx + 1}:**", doc_info["Metadata"].get("source", "Unknown"))
            #st.write("**내용 예시:**", doc_info["Content"])
    else:
        st.write("No documents found in the collection.")

def delete_documents():
    # Initialize Chroma PersistentClient
    db_file_path = DB_PATH  # Replace with your database file path
    # Check if the file exists and delete it
    if os.path.exists(db_file_path):
        os.remove(db_file_path)
        print("Database file deleted successfully.")
    else:
        print("Database file not found.")
    
    st.rerun()
