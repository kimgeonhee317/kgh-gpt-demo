#Conversation storage currently not working

import os
import json
import streamlit as st
from openai import OpenAI
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.storage.chat_store import SimpleChatStore
from global_settings import INDEX_STORAGE, CONVERSATION_FILE

def load_chat_store():
    try:
        chat_store = SimpleChatStore.from_persist_path(
            CONVERSATION_FILE
        )
    except FileNotFoundError:
        chat_store = SimpleChatStore()
    return chat_store

def display_messages(chat_store, container):
    with container:
        for message in chat_store.get_messages(key="0"):
            with st.chat_message(message.role):
                st.markdown(message.content)

def initialize_chatbot(user_name, goal, 
                       chat_store, container):
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000, 
        chat_store=chat_store, 
        chat_store_key="0"
    )  
    storage_context = StorageContext.from_defaults(
        persist_dir=INDEX_STORAGE
    )
    index = load_index_from_storage(
        storage_context, index_id="vector"
    )
    study_materials_engine = index.as_query_engine(
        similarity_top_k=3
    )
    study_materials_tool = QueryEngineTool(
        query_engine=study_materials_engine, 
        metadata=ToolMetadata(
            name="study_materials",
            description=(
                f"Provides official information about "
                f"{goal}. Use a detailed plain "
                f"text question as input to the tool."
            ),
        )
    )
    agent = OpenAIAgent.from_tools(
        tools=[study_materials_tool], 
        memory=memory,
        system_prompt=(
            f"당신은 챗 어시스턴트입니다. "
            f"당신의 목표는 {user_name}를 도와 "
            f"{goal} 업무를 수행하는 것입니다."
            f"한글에 대한 질문을 입력하면 영어로 번역된 답변을 제공합니다."
            f"반면 영어에 대한 질문은 한글로 번역된 답변을 제공합니다."
           # f"필요할 경우, 다음 배경 지식을 참고하세요: {context}"
        )
    )
    display_messages(chat_store, container)
    return agent

def chat_interface(agent, chat_store, container):  
    prompt = st.chat_input("입력")
    if prompt:
        with container:
            with st.chat_message("user"):
                st.markdown(prompt)
            response = str(agent.chat(prompt))
            with st.chat_message("assistant"):
                st.markdown(response)
        chat_store.persist(CONVERSATION_FILE)

