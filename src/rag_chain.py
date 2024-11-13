import os
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint
from langchain.schema.runnable import RunnableParallel
import time
import uuid
from uuid import uuid4
 
# Load the API key from env variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

system_prompt = (
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 원래 가지고있는 지식은 모두 배제하고, 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
"""
    "\n\n"
    "{context}"
)
 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# This function is used to format(concatenate) the page content of the documents
def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)




def create_rag_chain(chunks):

    # 임베딩 모델 정의
    clovax_embeddings = ClovaXEmbeddings(model='bge-m3')
    
    # 로컬 클라이언트 경로 지정
    client = chromadb.PersistentClient(path="./chroma_langchain_db") #저장할 로컬 경로
    
    # Chroma 컬렉션 생성
    chroma_collection = client.get_or_create_collection(
        name="clovastudiodatas_docs", #collection이 바뀔때마다 이름도 꼭 변경해줘야 합니다.
        metadata={"hnsw:space": "cosine"} #사용하는 임베딩 모델에 따라 ‘l2’, 'ip', ‘cosine’ 중에 사용
    )
    
    # Chroma 벡터 저장소 생성
    vectorstore = Chroma(
        client=client,
        collection_name="clovastudiodatas_docs",
        embedding_function=clovax_embeddings
    )
    
    # tqdm으로 for 루프 감싸기
    for doc in tqdm(chunks, desc="Adding documents", total=len(chunks)):
        # #print(doc)
        # doc_json = dict(text=doc.page_content)
        # #print(doc_json)
        # embeddings = embedding_executor.execute(doc_json)
        # #print(embeddings)
        embeddings = clovax_embeddings.embed_documents([doc.page_content])[0]
        # 문서 추가
        chroma_collection.add(
            ids=[str(uuid.uuid4())],  # 고유한 ID 생성
            documents=[doc.page_content],
            embeddings=[embeddings],
            metadatas=[doc.metadata]
        )
        time.sleep(1.1)  # 이용량 제어를 고려한 1초 이상의 딜레이, 필요에 따라 조정 가능
    
    print("All documents have been added to the vectorstore.")


    llm = ChatClovaX(
        model="HCX-003",
    )
    # 2. Chroma 컬렉션에 쿼리 실행하여 유사한 문서 검색

    
    # 3. 유사도 검색 결과를 사용하여 리트리버 구성
    similarity_retriever = vectorstore.as_retriever(
        kwargs={"k": 5},
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.1, "k": 3}
    )

    # 4. 검색된 문서를 RAG 체인에 연결하여 답변 생성
    rag_chain_from_docs = RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"])
    ) | prompt | llm | StrOutputParser()
    
    rag_chain_with_source = RunnableParallel(
        {"context": similarity_retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    pprint(rag_chain_with_source)