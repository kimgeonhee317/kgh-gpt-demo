import os
import langchain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX

# Load the API key from env variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

RAG_PROMPT_TEMPLATE = """
You are a helpful coding assistant that can answer questions about the provided context. The context is usually a PDF document or an image (screenshot) of a code file. Augment your answers with code snippets from the context if necessary.

If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
"""

RAG_PROMPT_TEMPLATE =  """[
        ("system", f"CLOVA Studio는 HyperCLOVA X 언어 모델을 활용하여 AI 서비스를 손쉽게 만들 수 있는 개발 도구입니다. 다음은 관련 컨텍스트입니다: {context}"),
        ("human", f"{question}")
    ]"""

PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# This function is used to format(concatenate) the page content of the documents
def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

def format_docs_and_capture_context(docs):
    # Process and format your documents here
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)  # Assuming docs are structured this way
    # Print or log the documents for debugging
    for doc in docs:
        print("\n\n## Retrieved Document:", doc.page_content)
    # You can also store these documents in a global or returning them in another way if needed
    return formatted_docs  # Return formatted context as a single string or as needed


def create_rag_chain(chunks):

    #model_name = "./embedding/bge-m3"
    #hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

    clovax_embeddings = ClovaXEmbeddings(model='bge-m3')

    #embeddings = OpenAIEmbeddings(api_key=api_key)
    doc_search = FAISS.from_documents(chunks, hf_embeddings)
    retriever = doc_search.as_retriever(
        search_type="similarity", search_kwargs={"k": 5} # k is the number of similar documents to retrieve
    )

    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = ChatClovaX(
        model="HCX-003" # 테스트 앱 또는 서비스 앱 인증 정보에 해당하는 모델명 입력 (기본값: HCX-003)
    )

    rag_chain = (
        {"context": retriever | format_docs_and_capture_context, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain