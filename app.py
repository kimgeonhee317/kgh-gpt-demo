__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import langchain
import os
import sys
import getpass
from dotenv import load_dotenv
from src.document_processor import process_document, show_chunk
from src.rag_chain import create_rag_chain


# check python version
#st.write("Python executable being used:", sys.executable)

# Load environment variables
load_dotenv()
langchain.verbose = True
st.set_page_config(page_title="NAVER HCX003 챗봇 데모버전(RAG 등 테스트용)", page_icon="🤖")

st.title("NAVER HCX003 chatbot demo v0.1\n")

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar for API key input
with st.sidebar:
    studio_key = st.text_input("Enter your CLOVA STUDIO Key", type="password")
    gw_key = st.text_input("Enter your CLOVA GW Key", type="password")
    embedding_id = st.text_input("Enter your embedding id", type="password")
    segmentation_id = st.text_input("Enter your segmentation id", type="password")
    if studio_key:
        os.environ["NCP_CLOVASTUDIO_API_KEY"] = studio_key
    if gw_key:
        os.environ["NCP_APIGW_API_KEY"] = gw_key
    if embedding_id:
        os.environ["NCP_CLOVASTUDIO_APP_ID"] = embedding_id
    if segmentation_id:
        os.environ["NCP_CLOVASTUDIO_APP_ID_SEGMENTATION"] = segmentation_id

    st.write("== UPDATE LOG ==")
    st.write("2021-11-13: demo version 0.1")

# local
if not studio_key:
    studio_key = os.getenv("NCP_CLOVASTUDIO_API_KEY")
if not gw_key:
    gw_key = os.getenv("NCP_APIGW_API_KEY")
if not embedding_id:
    embedding_id = os.getenv("NCP_CLOVASTUDIO_APP_ID")
if not segmentation_id:
    segmentation_id = os.getenv("NCP_CLOVASTUDIO_APP_ID_SEGMENTATION")

print(studio_key, gw_key, embedding_id, segmentation_id)

# File uploader
uploaded_file = st.file_uploader("RAG 파이프라인 구축을 위한 파일을 선택해 주세요.", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if st.button("벡터데이터베이스 생성 시작"):
        if studio_key and gw_key and embedding_id and segmentation_id:
            with st.spinner("벡터데이터베이스 생성 중입니다. 잠시만 기다려주세요..."):
                # Save the uploaded file temporarily
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    # Process the document
                    chunks = process_document(uploaded_file.name)
                    print(chunks[0])
                    show_chunk(chunks)
                    #   Create RAG chain
                    st.session_state.rag_chain = create_rag_chain(chunks)
                    print(st.session_state.rag_chain)
                    st.success("성공적으로 RAG 환경 구축이 완료되었습니다.")
                except ValueError as e:
                    st.error(str(e))
                finally:
                    # Remove the temporary file
                    os.remove(uploaded_file.name)
        else:
            if not studio_key:
                st.error("Please provide your CLOVA STUDIO API key.")
            elif not gw_key:
                st.error("Please provide your CLOVA GW API key.")
            elif not embedding_id:
                st.error("Please provide your embedding id.")
            elif not segmentation_id:
                st.error("Please provide your segmentation id.")
            else:
                st.error("Please provide all the required API keys.")

# Query input
query = st.text_input("Ask a question about the uploaded document")

if st.button("Ask"):
    if st.session_state.rag_chain and query:
        with st.spinner("Generating answer..."):
            result = st.session_state.rag_chain.invoke(query)

            st.subheader("Answer:")
            st.write(result)
            query = ""
    elif not st.session_state.rag_chain:
        st.error("Please upload and process a file first.")
    else:
        st.error("Please enter a question.")