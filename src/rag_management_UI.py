import streamlit as st
import os
from src.session_management import save_session
from src.global_settings import STORAGE_PATH
from src.rag_chain import create_rag_chain
from src.document_segmentation import process_document, show_chunk
from src.default_UI import show_default_UI


def show_ragmgmt_UI():

    st.title("RAG 파이프라인 구축")

    # Present currently set systemprompt
    default_prompt = (
            """당신은 다양한 문서에서 얻은 지식을 바탕으로 질문에 답할 수 있는 능력이 있는 지식인형 어시스턴트입니다. 응답할 때는 자신이 읽은 여러 문서들로부터 얻은 사실, 아이디어, 세부 정보가 포함된 내부 도서관을 갖고 있다고 생각하세요. 이 정보를 활용하여 정확하고 통찰력 있는, 상세한 답변을 제공하세요. 응답 시 정보를 자연스럽게 통합하되, 특정 문서를 직접 인용하거나 명시적으로 언급하는 것은 꼭 필요한 경우에만 하세요. 목표는 모든 지식을 당신의 이해에서 비롯된 것처럼 제공하여 자연스럽고 유익하게 대화를 이끌어가는 것입니다. 
            \n\n 주어진 문맥: {context}
            \n\n 질문의 본질에 집중하고 통합된 지식을 활용하여 대화를 심도 있고 정확하게 풍부하게 만드세요.
            """)
    if 'system_prompt' not in st.session_state:
        st.session_state['system_prompt'] = default_prompt
        
    user_prompt = st.text_area("(1) 기본 시스템 프롬프트 설정 (RAG를 통해 주입될 추가 정보는 \{context\}로 표현):", value=st.session_state.system_prompt, height=400)
    if st.button("변경"):
        st.session_state.system_prompt = user_prompt
        # complete
        st.success("시스템 프롬프트가 변경되었습니다.")


    # File uploader
    uploaded_file = st.file_uploader("(2) 벡터데이터베이스 구축을 위한 파일을 선택해 주세요.", type=["pdf", "png", "jpg", "jpeg"])

    # Initialize rag chain
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


    if uploaded_file is not None:
        if st.button("파이프라인 구축 시작"):
            with st.spinner("잠시만 기다려주세요..."):
                # Save the uploaded file temporarily
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    # Process the document
                    chunks = process_document(uploaded_file.name)
                    #print(chunks)
                    show_chunk(chunks)
                    #   Create RAG chain
                    st.session_state.rag_chain = create_rag_chain(chunks)
                    print(st.session_state.rag_chain)
                    st.success("성공적으로 파이프라인 환경 구축이 완료되었습니다.")
                except ValueError as e:
                    st.error(str(e))
                finally:
                    # Remove the temporary file
                    os.remove(uploaded_file.name)
                st.session_state

    if st.button("기본 화면으로"):
        st.session_state['rag_pipeline'] = False
        st.rerun()

    # # Query input
    # query = st.text_input("챗봇에게 질의하세요. (아직 히스토리 기능 미구현')")

    # if st.button("질의"):
    #     if st.session_state.rag_chain and query:
    #         with st.spinner("답변 생성 중..."):
    #             result = st.session_state.rag_chain.invoke(query)

    #             st.subheader("답변:")
    #             st.write(result)
    #             query = ""
    #     elif not st.session_state.rag_chain:
    #         st.error("파일을 생성해서 RAG를 먼저 구축하세요.")
    #     else:
    #         st.error("질의문이 필요합니다.")
