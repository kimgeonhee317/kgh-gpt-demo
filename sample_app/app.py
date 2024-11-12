from user_onboarding import user_onboarding
from session_functions import load_session, delete_session, save_session
from logging_functions import reset_log
from quiz_UI import show_quiz
from training_UI import show_training_UI
import streamlit as st
import os
from global_settings import CONVERSATION_FILE

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title('Chatbot adaption Demo for NAVER HCX')

    model = "HCX003"
    st.session_state['model'] = model

    st.sidebar.markdown(f'### 연결 모델: {model}')

    if 'OPENAI_API_KEY' not in st.session_state or not st.session_state['OPENAI_API_KEY']:
        #api_key = st.text_input("Enter your OpenAI API Key (or leave blank if running locally): ")
        api_key = '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'
        st.session_state['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_API_KEY'] = api_key

        

    # Debugging: Print the session state
    print(st.session_state)
    if os.path.exists(CONVERSATION_FILE):
        os.remove(CONVERSATION_FILE)

    # Check if the user is returning and has opted to take a quiz
    if 'resume_session' in st.session_state and st.session_state['resume_session']:
        #st.write(f"대화를 시작합니다!")
        # If resuming, clear previous content and show the training UI
        show_training_UI(st.session_state['user_name'], st.session_state['goal'])
    elif not load_session(st.session_state): # Check if the user is new
        user_onboarding()  # Show the onboarding screen for new users
    else:
        # For returning users, display options to resume or start a new session
        st.write(f"다시 오신 것을 환영합니다. {st.session_state['user_name']}!")
        col1, col2 = st.columns(2)
        if col1.button(f"챗봇을 재개합니다"):
            # Mark the session to be resumed and rerun to clear previous content
            st.session_state['resume_session'] = True
            st.rerun()
        if col2.button('다시 초기 설정을 시작합니다'):
            delete_session(st.session_state)
            reset_log()
            # Clear session state and rerun for a fresh start
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
