from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatClovaX


def create_rewritten_query(query: str):
    llm = get_rewriting_llm()
    query_rewriter_chain = build_query_rewriting_chain(llm)
    rewritten_query = rewrite_query(query, query_rewriter_chain)
    print("Original query:", query)
    print("\nRewritten query:", rewritten_query)
    return rewritten_query

# LLM configuration
def get_rewriting_llm(model_name: str = "HCX-003", temperature: float = 0.5, max_tokens: int = 1024, repeat_penalty=5):
    return ChatClovaX(temperature=temperature, model_name=model_name, max_tokens=max_tokens)

# Prompt Template for Query Rewriting
def create_query_rewrite_prompt() -> PromptTemplate:
    template = """
    당신은 RAG 시스템에서 검색을 개선하기 위해 사용자 쿼리를 재구성하는 임무를 맡은 AI 어시스턴트입니다.
    원래 쿼리를 고려하여 관련 정보를 검색할 가능성이 높고, 더 구체적이고, 세부적이면서 한국은행에 특화된 쿼리를 다시 작성합니다.

    원 쿼리: {original_query}

    재작성된 쿼리:
    """
    return PromptTemplate(input_variables=["original_query"], template=template)

# Build Query Rewriting Chain
def build_query_rewriting_chain(llm: ChatClovaX) -> LLMChain:
    prompt_template = create_query_rewrite_prompt()
    return prompt_template | llm

# Function to Rewrite the Query
def rewrite_query(original_query: str, query_rewriter_chain: LLMChain) -> str:
    response = query_rewriter_chain.invoke(original_query)
    return response.content.strip()