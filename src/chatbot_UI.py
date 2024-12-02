import streamlit as st
import os
from src.session_management import save_session
from src.global_settings import STORAGE_PATH
from src.rag_chain_multi_turn import  get_rag_chain
from src.utils import stream_response
from src.query_rewriting import create_rewritten_query
def initialize_rag_chain():
    if st.session_state.get('rag_chain') is None:
        with st.spinner("Initializing RAG pipeline..."):
            st.session_state['rag_chain'] = get_rag_chain()
            st.session_state.urls = []
            st.session_state.source_contents = []
            st.session_state.rewrite = False

def display_system_prompt():
    default_prompt = (
            """당신은 다양한 문서에서 얻은 지식을 바탕으로 질문에 답할 수 있는 능력이 있는 한국은행(Bank of Korea) 임직원을 위한 지식인형 어시스턴트입니다. 응답할 때는 자신이 읽은 여러 문서들로부터 얻은 사실, 아이디어, 세부 정보가 포함된 내부 도서관을 갖고 있다고 생각하세요. 이 정보를 활용하여 정확하고 통찰력 있는, 상세한 답변을 제공하세요. 응답 시 정보를 자연스럽게 통합하되, 특정 문서를 직접 인용하거나 명시적으로 언급하는 것은 꼭 필요한 경우에만 하세요. 목표는 모든 지식을 당신의 이해에서 비롯된 것처럼 제공하여 자연스럽고 유익하게 대화를 이끌어가는 것입니다. 
            \n\n 주어진 문맥: {context}
            \n\n 질문의 본질에 집중하고 통합된 지식을 활용하여 대화를 심도 있고 정확하게 풍부하게 만드세요.
            """)

    if 'system_prompt' not in st.session_state:
        st.session_state['system_prompt'] = default_prompt
    st.session_state.system_prompt = st.session_state.get('system_prompt', default_prompt)

def handle_query():
    query = st.session_state.query
    if query:
        if st.session_state.rewrite:
            # Rewrite the query
            r_query = create_rewritten_query(query = query)
        else:
            r_query = query
        # Append the user query to the conversation history
        st.session_state.messages.append(("human", r_query))
        with st.spinner("답변 생성 중..."):
            # Invoke the RAG chain
            st.session_state.rag_chain = get_rag_chain()
            response = st.session_state.rag_chain.invoke(r_query)
            naive_stream = st.session_state.rag_chain.stream(query)
            # # 스트리밍 출력
            # for chunk in naive_stream:
            #     print(f"Type: {type(chunk)}, Content: {chunk}")
            # Append the AI response to the conversation history
            st.session_state.messages.append(("ai", response['answer']))
            # Store the query and response for display
            st.session_state.responses.append((query, response['answer']))
            st.session_state.query = ""  # Clear the input for the next message

            # rag source
            print(f"raw response context {response['context']}")
            if len(response['context']) == 0:
                st.session_state.urls.append([])
            else:
                st.session_state.urls.append(list(response['context']))
            #print(st.session_state.urls)
            # st.session_state.source_contents.append(list({doc.page_content for doc in response['context']}))
            # print(st.session_state.source_contents)
    else:
        st.error("Please enter a query.")

def show_chatbot_UI():
    display_system_prompt()
    initialize_rag_chain()
    # Initialize conversation history if not set
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    if 'messages' not in st.session_state:
        st.session_state.messages = [("system", st.session_state.system_prompt)]

    st.title("챗봇 대화하기")
    rewrite = st.checkbox("Rewrite")
    st.session_state.rewrite = rewrite

    # Display conversation history
    with st.container():
        for idx, (query, response) in enumerate(st.session_state.responses):
            st.markdown(f"**You:** {query}")
            st.markdown(f"**Assistant:** {response}")

            if len(st.session_state.urls[idx]) == 0:
                st.info(f'**검색된 문서 없음**')
            else:
                #st.info(f'**전체** {temp_source_contents}')
                for i in range(len(st.session_state.urls[idx])):
                    st.info(f'**검색된 문서 {i+1}:** {st.session_state.urls[idx][i].metadata} \n\n **검색된 내용 {i+1}:** {st.session_state.urls[idx][i].page_content} \n\n')
                # for i in range(len(st.session_state.source_contents[idx])):
                #     st.info(f"**검색된 문서 내용 {i+1}:** {st.session_state.source_contents[idx][i]}")
                    
        if st.session_state.responses:
            st.session_state.last_query = st.session_state.responses[-1][0]  # Scroll to last query

    # Input area
    st.text_input("Message to Chatbot:", key="query", on_change=handle_query, placeholder="Type your query and press enter...")

    st.warning(f"**Current System Prompt:**\n\n{st.session_state.system_prompt}")


    st.write('---')
    if st.button("대화 초기화"):
        st.session_state.responses = []
        st.session_state.messages = [("system", st.session_state.system_prompt)]
        st.session_state.urls = []
        st.session_state.source_contents = []
        st.rerun()
    user_prompt = st.text_area("시스템 프롬프트 변경 (RAG를 통해 주입될 추가 정보는 \{context\}로 표현):", value=st.session_state.system_prompt, height=400)
    if st.button("시스템 프롬프트 변경"):
        st.session_state.system_prompt = user_prompt
        # complete
        st.success("시스템 프롬프트가 변경되었습니다. 대화를 초기화하여야만 적용됩니다.")
        st.rerun()

    st.write('---')
    if st.button("기본 화면으로"):
        st.session_state['run_chatbot'] = False
        st.rerun()

    st.write("\n\n(참고) 현재 저장된 세션 데이터는 다음과 같습니다.")
    st.write(st.session_state)

