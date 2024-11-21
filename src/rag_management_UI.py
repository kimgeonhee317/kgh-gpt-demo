import streamlit as st
import os
from src.session_management import save_session
from src.global_settings import STORAGE_PATH
from src.rag_chain import create_rag_chain
from src.document_segmentation import process_document, show_chunk
from src.vector_database_creator import create_vector_database
from src.default_UI import show_default_UI
from src.db_access import get_embedded_documents, delete_documents


def show_ragmgmt_UI():


    st.title("RAG 파이프라인 구축")

    st.write("---"*40)
    st.write("(참고) 현재 벡터데이터베이스에 저장된 자료는 다음과 같습니다.")
    get_embedded_documents()

    if st.button("DB 초기화"):
        st.write("벡터데이터베이스를 초기화합니다.")
        delete_documents()
        st.write("벡터데이터베이스 초기화 완료.")

    st.write("---"*40)



    # File uploader
    uploaded_files = st.file_uploader("벡터데이터베이스 구축을 위한 파일을 선택해 주세요.", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    finish_upload = st.button('업로드 및 파이프라인 구축 시작')

    if uploaded_files and finish_upload:
        if not os.path.exists(STORAGE_PATH):
            os.makedirs(STORAGE_PATH)  # Create directory if it does not exist
            st.write(f"Created directory at {STORAGE_PATH}")

        with st.spinner("잠시만 기다려주세요..."):
            saved_file_names = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(STORAGE_PATH, uploaded_file.name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_file_names.append(uploaded_file.name)
                    try:
                        chunks = process_document(file_path)  # Ensure this handles local paths
                        print("\n샘플 청크 (처음 3개):")
                        for i, chunk in enumerate(chunks[:3], 1):
                            print(f"\n청크 {i}:")
                            print(f"메타데이터: {chunk['metadata']}")
                            print(f"내용: {chunk['page_content']}")
                            print(f"길이: {len(chunk['page_content'])} 문자")
                        st.success("파일 분할 완료: " + str(uploaded_file.name))
                        create_vector_database(chunks)
                        st.success("벡터 데이터베이스 저장 완료: " + str(uploaded_file.name))
                    except ValueError as e:
                        st.error(str(e))
                except IOError as e:
                    st.error(f"파일 저장 실패: {e}")

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
