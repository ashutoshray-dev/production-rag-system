from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv
load_dotenv()
import os
from langsmith import traceable
from langchain_core.runnables import RunnableConfig
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import pickle
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
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=os.getenv("GOOGLE_API_KEY"), streaming=True)
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
