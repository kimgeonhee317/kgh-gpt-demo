import logging
import http.client
import os
import json
import uuid
from uuid import uuid4
import streamlit as st
from tqdm import tqdm
from http import HTTPStatus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers.pdf import (
   extract_from_images_with_rapidocr,
)
from langchain.schema import Document

class CLOVAStudioExecutor:
    def __init__(self, host, request_id=None):
        self._host = host
        self._api_key = os.environ["NCP_CLOVASTUDIO_API_KEY"]
        self._api_key_primary_val = os.environ["NCP_APIGW_API_KEY"]
        self._request_id = request_id or str(uuid.uuid4())
  
    def _send_request(self, completion_request, endpoint):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }
  
        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', endpoint, json.dumps(completion_request), headers)
        response = conn.getresponse()
        status = response.status
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result, status
  
    def execute(self, completion_request, endpoint):
        res, status = self._send_request(completion_request, endpoint)
        if status == HTTPStatus.OK:
            return res, status
        else:
            error_message = res.get("status", {}).get("message", "Unknown error") if isinstance(res, dict) else "Unknown error"
            raise ValueError(f"오류 발생: HTTP {status}, 메시지: {error_message}")
  
class SegmentationExecutor(CLOVAStudioExecutor):
    def execute(self, completion_request):
        app_id = os.environ["NCP_CLOVASTUDIO_APP_ID_SEGMENTATION"]
        endpoint = f'/testapp/v1/api-tools/segmentation/{app_id}'
        res, status = super().execute(completion_request, endpoint)
        if status == HTTPStatus.OK and "result" in res:
            return res["result"]["topicSeg"]
        else:
            error_message = res.get("status", {}).get("message", "Unknown error") if isinstance(res, dict) else "Unknown error"
            raise ValueError(f"오류 발생: HTTP {status}, 메시지: {error_message}")
         

def process_pdf(source):
    loader = PyPDFLoader(source)
    documents = loader.load()

    required_env_vars = [
        "NCP_CLOVASTUDIO_API_KEY",
        "NCP_APIGW_API_KEY",
        "NCP_CLOVASTUDIO_APP_ID_SEGMENTATION"
    ]   

    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


   # Filter out scanned pages
    unscanned_documents = [doc for doc in documents if doc.page_content.strip() != ""]

    scanned_pages = len(documents) - len(unscanned_documents)
    if scanned_pages > 0:
        logging.info(f"Omitted {scanned_pages} scanned page(s) from the PDF.")

    if not unscanned_documents:
        raise ValueError(
            "All pages in the PDF appear to be scanned. Please use a PDF with text content."
        )
    
    return split_documents(unscanned_documents)

def process_image(source):
   # Extract text from image using OCR
   with open(source, "rb") as image_file:
       image_bytes = image_file.read()

   extracted_text = extract_from_images_with_rapidocr([image_bytes])
   documents = [Document(page_content=extracted_text, metadata={"source": source})]
   return split_documents(documents)

def split_documents(documents):
   # Split documents into smaller chunks for processing
   text_splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
   )
   return text_splitter.split_documents(documents)


def segment_documents(documents):
    print(documents)
    segmentation_executor = SegmentationExecutor(
        host="clovastudio.apigw.ntruss.com"
    )

    chunked_data = []

    for data in tqdm(clovastudiodatas_flattened):
        try:
            request_data = {
                "postProcessMaxSize": 100,
                "alpha": -100,
                "segCnt": -1,
                "postProcessMinSize": -1,
                "text": data.page_content,
                "postProcess": True
            }
              
            response_data = segmentation_executor.execute(request_data)
            result_data = [' '.join(segment) for segment in response_data]
      
            for paragraph in result_data:
                chunked_document = {
                    "metadata": data.metadata["source"],
                    "page_content": paragraph
                }
                chunked_data.append(chunked_document)
      
        except Exception as e:
            print(f"Error processing data from {data.metadata['source']}: {e}")
            # 오류 발생 시 현재 반복을 건너뛰고 다음으로 진행
            continue
   
    print(len(chunked_data))
    
    return segment_documents(documents)


def process_document(source):
   # Determine file type and process accordingly
   if source.lower().endswith(".pdf"):
       return process_pdf(source)
   elif source.lower().endswith((".png", ".jpg", ".jpeg")):
       return process_image(source)
   else:
       raise ValueError(f"Unsupported file type: {source}")
   
def show_chunk(documents):
    st.write(f"10개만 출력합니다.")
    for i, doc in enumerate(documents[:10]):
        print(f"Document {i}:")
        print(f"  Page Content: {doc.page_content[:100]}...")  # 첫 100자만 출력
        print(f"  Metadata: {doc.metadata}")
        print(f"  Page Content Type: {type(doc.page_content)}")
        print(f"  Metadata Type: {type(doc.metadata)}")
        print("-" * 40)

        st.write(f"Document {i}:")
        st.write(f"  Page Content: {doc.page_content[:100]}...")
        st.write(f"  Metadata: {doc.metadata}")
        st.write(f"  Page Content Type: {type(doc.page_content)}")
        st.write(f"  Metadata: {type(doc.metadata)}")
        st.write("-" * 40)