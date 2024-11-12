import streamlit as st
from slides import Slide, SlideDeck
import json
import os
from global_settings import CONVERSATION_FILE
from openai import OpenAI
from pathlib import Path
from conversation_engine import initialize_chatbot, chat_interface, load_chat_store

def show_training_UI(user_name, goal):
    # Load the slide deck
    #slide_deck = SlideDeck.load_from_file("cache/slides.json")
    
    # Display title and slide navigation controls
    # st.sidebar.markdown("## " + slide_deck.topic)
    # current_slide_index = st.sidebar.number_input("Slide Number", min_value=0, max_value=len(slide_deck.slides)-1, value=0, step=1)
    # current_slide = slide_deck.slides[current_slide_index]
    # if st.sidebar.button("Toggle narration"):
    #         st.session_state.show_narration = not st.session_state.get('show_narration', False)

    # Displaying slides and narration in the main area
    # col1, col2 = st.columns([0.7,0.3],gap="medium")
    # with col1:
    #     st.markdown(current_slide.render(display_narration=st.session_state.get('show_narration', False)), unsafe_allow_html=True)

    # Chatbot integration in the sidebar
    # with col2:

    st.header("BOK Chatbot Agent")
    st.success(f"안녕하세요 {user_name}님. 저는 다음과 같은 목표를 수행합니다: {goal}")

    st.write(st.session_state)
    #with st.spinner("Preparing the chatbot..."):
    chat_store = load_chat_store()
    container = st.container(height=300)
    agent = initialize_chatbot(user_name, goal, chat_store, container)
    chat_interface(agent, chat_store, container)

    if st.button("End Session and Clear Conversation"):
        if os.path.exists(CONVERSATION_FILE):
            os.remove(CONVERSATION_FILE)
            st.success("Conversation file deleted.")
            st.rerun()
        else:
            st.warning("No conversation file found to delete.")

