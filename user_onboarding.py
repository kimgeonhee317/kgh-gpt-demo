import streamlit as st
import os
from session_functions import save_session
from logging_functions import log_action
from global_settings import STORAGE_PATH
from document_uploader import ingest_documents
from training_material_builder import generate_slides
from index_builder import build_indexes
from quiz_builder import build_quiz
import pickle
import pandas as pd

def user_onboarding():
    st.write(f"반갑습니다. 한국은행 HCX 기반 챗봇 데모 버전입니다. 초기 설정을 시작합니다.")

    user_name = st.text_input('이름을 입력해주세요:',)
    if not user_name: return

    st.session_state['user_name'] = user_name
    st.write(f"안녕하세요 {user_name}. 반갑습니다.")
    
    goal = st.text_input('사용 목적을 입력해주세요:',)
    if not goal: return
    st.session_state['goal'] = goal

    print(st.session_state)
    st.write(f"다음과 같은 작업을 수행합니다: '{goal}'")

    if goal:
        st.write("참고할 문서가 있으시면 업로드해주세요.")
        uploaded_files = st.file_uploader("파일 선택", accept_multiple_files=True)
        finish_upload = st.button('업로드')

        if finish_upload and uploaded_files:
            saved_file_names = []
            st.info('업로드 중입니다...')
            for uploaded_file in uploaded_files:
                file_path = os.path.join(STORAGE_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_file_names.append(uploaded_file.name)
                st.write(f"다음 파일이 정상적으로 업로드되었습니다: {uploaded_file.name}")

            st.session_state['uploaded_files'] = saved_file_names
            st.session_state['finish_upload'] = True

    st.write(f"업로드 자료를 반영합니다...")
    if 'finish_upload' in st.session_state:
        save_session(st.session_state)
        log_action(
            f"사용 목적 : {goal}",
            action_type="ONBOARDING"
        )
        st.info('참고 문서 로딩 중...')
        nodes = ingest_documents()
        st.info('참고 문서 인덱싱 중...')
        vector_index , tree_index = build_indexes (nodes)

        st.write('인덱싱이 완료되었습니다.')
        #generate_slides(goal)
    
        st.session_state['resume_session'] = True
        st.rerun()
