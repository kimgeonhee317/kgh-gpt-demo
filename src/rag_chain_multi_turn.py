import os
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint
from langchain.schema.runnable import RunnableParallel
from pydantic import ConfigDict

import time
import uuid
from uuid import uuid4
import streamlit as st


# Load the API key from env variables
load_dotenv()
chromadb.api.client.SharedSystemClient.clear_system_cache()


def add_message(role, content):
    st.session_state.messages.append((role, content))

def get_rag_chain():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            ("system", st.session_state.system_prompt)  # Initial system message
        ]

    prompt = ChatPromptTemplate.from_messages(st.session_state.messages)
    prompt.pretty_print() # Print the prompt

    # 임베딩 모델 정의
    clovax_embeddings = ClovaXEmbeddings(model='bge-m3')
    ClovaXEmbeddings.Config.protected_namespaces = ()

    # 로컬 클라이언트 경로 지정
    client = chromadb.PersistentClient(path="./chroma_langchain_db", settings=Settings(anonymized_telemetry=False)) # 저장할 로컬 경로
    
    # Chroma 벡터 저장소 생성 (기존 컬렉션에 연결)
    vectorstore = Chroma(
        client=client,
        collection_name="clovastudiodatas_docs",
        embedding_function=clovax_embeddings
    )
    
    # Chroma retriever 생성 (기존 벡터 저장소를 이용한 검색)
    retriever = vectorstore.as_retriever(
        kwargs={"k": 5},
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3}
    )

    print("Vectorstore is now accessible for retrieval only.")

    # Define the LLM
    llm = ChatClovaX(
        model="HCX-003",
        max_tokens=1024,
        temperature=0.5,
        repeat_penalty=5
    )

    # Define the RAG chain
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_with_metadata(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    # This function is used to format(concatenate) the page content of the documents
    def format_docs(docs): 
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_with_metadata(docs):
        """
        Format documents by combining their content with metadata (source name).
        """
        formatted_docs = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown source")
            formatted_docs.append(f"Source: {source}\nContent: {doc.page_content}")
        return "\n\n".join(formatted_docs)

        
    return rag_chain_with_source
