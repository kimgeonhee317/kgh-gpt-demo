__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import langchain
import os
import sys
import getpass
from dotenv import load_dotenv
#from src.document_processor import process_document, show_chunk
from src.document_segmentation import process_document, show_chunk
from src.session_management import load_session, delete_session, save_session
from src.default_UI import show_default_UI
from src.onboarding import execute_onboarding
from src.chatbot_UI import show_chatbot_UI
from src.rag_management_UI import show_ragmgmt_UI

def main():
    # Load environment variables
    load_dotenv()
    langchain.verbose = True
    st.set_page_config(page_title="NAVER HCX003 챗봇 데모버전(RAG 등 테스트용)", page_icon="🤖", layout="wide")
    st.title("NAVER HCX003 chatbot demo v0.2\n")

    # Sidebar for API key input management and update log
    with st.sidebar:
        st.sidebar.title('kgh-gpt-test')
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
        st.write("2024-11-13: initial version")
        st.write("2024-11-18: add HCX segmentator to the pipeline")
        st.write("2024-11-19: add Session management, multi-turn RAG chatbot")
        st.write("2024-11-20: demo v0.2: present RAG retrieved source for each turn")
        st.write("2024-12-3: query rewriting feature is added")

    # (!) only for the local
    if not studio_key:
        studio_key = os.getenv("NCP_CLOVASTUDIO_API_KEY")
    if not gw_key:
        gw_key = os.getenv("NCP_APIGW_API_KEY")
    if not embedding_id:
        embedding_id = os.getenv("NCP_CLOVASTUDIO_APP_ID")
    if not segmentation_id:
        segmentation_id = os.getenv("NCP_CLOVASTUDIO_APP_ID_SEGMENTATION")
    #print(studio_key, gw_key, embedding_id, segmentation_id)

    if not studio_key or not gw_key or not embedding_id or not segmentation_id:
        st.warning("API 키를 입력해주세요.")
        return
    
    # Session Setup    
    if "user_name" not in st.session_state:
        if load_session(st.session_state):
            st.rerun()
        else:
            st.session_state.user_name = None
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = False
    if "run_chatbot" not in st.session_state:
        st.session_state.run_chatbot = False
   # Initialize rag chain
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    # Session Routing
    if st.session_state.user_name is None:
        execute_onboarding()
    elif st.session_state['rag_pipeline'] == True:
        show_ragmgmt_UI()
    elif st.session_state['run_chatbot'] == True:
        show_chatbot_UI()
    else: 
        show_default_UI()

if __name__ == "__main__":
    main()