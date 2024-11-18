import streamlit as st
import os
from src.session_management import save_session
from src.global_settings import STORAGE_PATH
import pandas as pd

def execute_onboarding():

    studio_key = os.environ["NCP_CLOVASTUDIO_API_KEY"]
    gw_key = os.environ["NCP_APIGW_API_KEY"]
    embedding_id = os.environ["NCP_CLOVASTUDIO_APP_ID"]
    segmentation_id = os.environ["NCP_CLOVASTUDIO_APP_ID_SEGMENTATION"]

    # make directory for the session if needed
    user_name = st.text_input('세션관리를 위해 사용자명을 입력해주세요.')
    if not user_name: return
    proceed_button = st.button('저장')
    if proceed_button:
        # key check
        if studio_key and gw_key and embedding_id and segmentation_id:
            if 'user_name' not in st.session_state:
                st.session_state['user_name'] = None

            st.session_state['user_name'] = user_name
            save_session(st.session_state)
            st.rerun()
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
                st.error("Please provide all the necessary keys.")

    