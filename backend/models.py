from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

os.environ["LANGCHAIN_PROJECT"] = "Production-RAG-system"

# <------------------- pydantic objects ----------------------->
class ChatTitle(BaseModel):
    chat_title: str = Field(description="A brief 4-5word title that captures the essence of the input.")


class FinalResponse(BaseModel):
    answer: str = Field(description="The answer with inline citations in format: [page no:X, source:filename]")


# <------------------ model ------------------------->
# model = ChatOllama(model='qwen2.5:3b')
# model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=os.getenv("GOOGLE_API_KEY"), streaming=True)
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=os.getenv("GOOGLE_API_KEY"), streaming=True)
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
