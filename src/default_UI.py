import streamlit as st
from src.session_management import save_session, delete_session
from src.global_settings import STORAGE_PATH

def show_default_UI():

    if not st.session_state['user_name']:
        st.write("세션 데이터가 손실되었습니다. 다시 시작해주세요.")
        delete_session(st.session_state)
        st.rerun()

    st.write(f"환영합니다, {st.session_state.user_name}님!")
    st.write("원하시는 작업을 선택하세요.")

    col1, col2, col3 = st.columns(3)
    if col1.button('(1) RAG Pipeline 관리'):
        # Mark the session to be resumed and rerun to clear previous content
        st.session_state['rag_pipeline'] = True
        st.rerun()
    elif col2.button('(2) 챗봇 시작'):
        st.session_state['run_chatbot'] = True
        st.rerun()

    elif col3.button('(3) 세션 삭제'):
        st.session_state.clear()
        delete_session(st.session_state)
        st.write("세션 데이터가 삭제되었습니다.")
        st.rerun()
    
    st.write("(참고) 현재 저장된 세션 데이터는 다음과 같습니다.")
    st.write(st.session_state)
