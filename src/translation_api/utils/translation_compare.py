# connect to milvus db

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
from src.translation_api.config import get_settings
load_dotenv()
settings = get_settings()

MILVUS_URI = settings.milvus_uri 
MILVUS_TOKEN = settings.milvus_token 
MILVUS_COLLECTION_NAME = settings.milvus_collection_name

EMBEDDING_MODEL ="models/embedding-001"
GEMINI_MODEL = "gemini-2.5-flash-lite"

milvus_client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
    collection_name=MILVUS_COLLECTION_NAME
)

import os
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY,
)

def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using Gemini."""
    return embeddings.embed_query(text)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class TranslationWithScoreAndExplanation(BaseModel):
    translation: str = Field(description="The translated text. this should be a single translation not a list of translations , should be the source text")
    score: float = Field(description="Score evaluating the translation quality, between 0 and 100.")
    explanation: str = Field(description="Explanation of the given score for this translation.")
    
class TranslationList(BaseModel):
    items: List[TranslationWithScoreAndExplanation]
    
output_parser = StrOutputParser()

structured_llm_grader = llm.with_structured_output(Grade)
structured_llm_translation_with_score_and_explanation = llm.with_structured_output(TranslationList)

def askgemini(prompt: str,is_boolean:bool ) -> str:
    """Generate a response from Gemini for a given prompt, using LangChain output parser."""
    if is_boolean:
        score = structured_llm_grader.invoke(prompt)
        return score.binary_score=="yes"
    else:
        response = llm.invoke(prompt)
    # Use output parser to ensure string output
    return response


# search a text in milvus

from pymilvus import AnnSearchRequest, RRFRanker



def search_text(milvus_client: MilvusClient, text: str):
    """
    search a text in milvus
    """
    query_embedding=get_embedding(text)
    results = milvus_client.search(
    collection_name=MILVUS_COLLECTION_NAME,
    data=[query_embedding],
    limit=5,
    output_fields=["text"],
    anns_field="dense_vector"
    )
    
    return results

def hybrid_search(milvus_client: MilvusClient, text: str):
    query_embedding=get_embedding(text)
    limit=5
    search_param_1 = {
        "data": [query_embedding],
        "anns_field": "dense_vector",
        "param": {},
        "limit": limit,
    } 
    request_1 = AnnSearchRequest(**search_param_1)
    
    search_param_2 = {
        "data": [query_embedding],
        "anns_field": "dense_vector",
        "param": {"drop_ratio_search": 0.2},
        "limit": limit
    }
    request_2 = AnnSearchRequest(**search_param_2)
    
    
    results = milvus_client.hybrid_search(
    collection_name=MILVUS_COLLECTION_NAME,
    vector=[query_embedding],
    reqs=[request_1,request_2],
    ranker=RRFRanker(),
    limit=limit,
    output_fields=["text","parent_id"]
)
    return results





# Generate translation using Gemini with context from relevant text
def generate_translation_with_context(source_text, context):
    prompt = f"Using the following context, translate the Tibetan text to English.\n\n 1.do not include any helping text only response with the translation\n\n 2. do not include the context in the translation only translate the text source text\n\n Context:\n {context}\n source text:\n {source_text}\n\n Translation:"
    response = askgemini(prompt,False)
    return response

def check_if_context_is_relevant(context,text_to_translate):
    prompt = f"Using the following context, check if the context is related to text in any sense.\n\n 1.do not include any helping text only response with the translation\n\n 2. do not include the context in the translation only translate the text source text\n\n 3. only response with true or false if false then reason why it is not relevant\n\n Context:\n {context}\n source text:\n {text_to_translate}\n\n Translation:"
    response = askgemini(prompt,True)
    return response




# compare the two translations with a score

def compare_translations(text_to_translate,translation_with_context, translation_without_context):
    prompt = f"Compare the following two translations and give a score between 0 and 100.\n\n source text:\n {text_to_translate}\n\n Translation with context:\n {translation_with_context}\n\n Translation without context:\n {translation_without_context}\n\n Score:"
    response = structured_llm_translation_with_score_and_explanation.invoke(prompt)
    return response



from pydantic import BaseModel

class TranslationComparisonResult(BaseModel):
    translation_with_context: str
    translation_without_context: str
    comparison_score: int
    explanation: str
    items: dict = None  # For additional fields if needed

def get_translation_with_context(text_to_translate: str) -> TranslationComparisonResult:
    # Get relevant text from Milvus
    print("text_to_translate",text_to_translate)
    relevant_results = hybrid_search(milvus_client, text_to_translate)
    print("relevant_results",relevant_results)
    relevant_texts = [hit.entity.get("text", "") for hit in relevant_results[0]]
    context = "\n\n".join(relevant_texts)
    print("context",context)
    translation_with_context = generate_translation_with_context(text_to_translate, context)
    translation_without_context = generate_translation_with_context(text_to_translate, "")
    print("translation_with_context",translation_with_context)
    print("translation_without_context",translation_without_context)
    translation_compare = compare_translations(text_to_translate,
        translation_with_context.content, 
        translation_without_context.content
    )
    # Assume translation_compare returns an object with score, explanation, and possible .items
    return translation_compare