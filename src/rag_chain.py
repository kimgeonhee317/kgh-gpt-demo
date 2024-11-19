import os
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
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

# This function is used to format(concatenate) the page content of the documents
def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.system_prompt),
            ("human", "{question}"),
        ]
    )
    # 임베딩 모델 정의
    clovax_embeddings = ClovaXEmbeddings(model='bge-m3')
    ClovaXEmbeddings.Config.protected_namespaces = ()

    # 로컬 클라이언트 경로 지정
    client = chromadb.PersistentClient(path="./chroma_langchain_db") # 저장할 로컬 경로
    
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
        max_tokens=2048,
        temperature=0.5,
        repeat_penalty=5
    )

    # Define the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def create_rag_chain(chunks):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.system_prompt),
            ("human", "{question}"),
        ]
    )

    # 임베딩 모델 정의
    clovax_embeddings = ClovaXEmbeddings(model='bge-m3')
    ClovaXEmbeddings.Config.protected_namespaces = ()

    # 로컬 클라이언트 경로 지정
    client = chromadb.PersistentClient(path="./chroma_langchain_db") #저장할 로컬 경로
    
    # Chroma 컬렉션 생성
    chroma_collection = client.get_or_create_collection(
        name="clovastudiodatas_docs", #collection이 바뀔때마다 이름도 꼭 변경해줘야 합니다.
        metadata={"hnsw:space": "cosine"} #사용하는 임베딩 모델에 따라 ‘l2’, 'ip', ‘cosine’ 중에 사용
    )
    
    # Chroma 벡터 저장소 생성
    vectorstore = Chroma(
        client=client,
        collection_name="clovastudiodatas_docs",
        embedding_function=clovax_embeddings
    )
    
    # tqdm으로 for 루프 감싸기
    for doc in tqdm(chunks, desc="Adding documents", total=len(chunks)):
        # #print(doc)
        # doc_json = dict(text=doc.page_content)
        # #print(doc_json)
        # embeddings = embedding_executor.execute(doc_json)
        # #print(embeddings)
        embeddings = clovax_embeddings.embed_documents([doc.page_content])[0]
        # 문서 추가
        chroma_collection.add(
            ids=[str(uuid.uuid4())],  # 고유한 ID 생성
            documents=[doc.page_content],
            embeddings=[embeddings],
            metadatas=[doc.metadata]
        )
        time.sleep(1.1)  # 이용량 제어를 고려한 1초 이상의 딜레이, 필요에 따라 조정 가능
    
    retriever = vectorstore.as_retriever(
        kwargs={"k": 5},
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3}
    )
    print("All documents have been added to the vectorstore.")

    llm = ChatClovaX(
        model="HCX-003",
        max_tokens = 2048,
        temperature= 0.5,
        repeat_penalty= 5
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain