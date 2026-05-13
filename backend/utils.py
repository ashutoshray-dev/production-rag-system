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



@traceable(name='load_file')
def load_file(path:str):
    document = PyPDFLoader(path)
    docs = document.load()
    return docs
@traceable(name='split_file')
def split_file(docs, chunk_size:int, chunk_overlap:int):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    return split_docs
# @traceable(name='build_store')
# def build_store(split_docs):
#     # embedding_model = OllamaEmbeddings(model='embeddinggemma')
#     embed_docs = FAISS.from_documents(split_docs, embedding_model)
#     return embed_docs
@traceable(name='setup_pipeline')
def setup_pipeline(path:str, chunk_size:int, chunk_overlap:int):
    docs = load_file(path=path)
    split_docs = split_file(docs=docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # embed_docs = build_store(split_docs=split_docs)
    return split_docs
@traceable(name='reranking')
def rerank(retrieved_docs:Document, query:str):
    sentence_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = encoder_model.predict(sentence_pairs)
    for i, doc in enumerate(retrieved_docs):
        doc.metadata["rerank_score"] = float(scores[i])
    reranking = retrieved_docs.sort(key=lambda x:x.metadata["rerank_score"], reverse=True)
    return retrieved_docs




def retrieve_threads_list():
    seen_threads = set()
    all_threads = list()
    for checkpoint in checkpointer.list(None):
        thread = checkpoint.config['configurable']['thread_id']
        if thread not in seen_threads:
            seen_threads.add(thread)
            all_threads.insert(0, thread)
    return all_threads
def generate_title(user_input):
    structured_model = model.with_structured_output(ChatTitle)
    title = structured_model.invoke(f'generate a suitable 4-5words title for this input that feels appropriate for the topic. If no input is present genetrate a random string.\ninput:{user_input}')
    # for models that return structured output with typeddict
    # return title['chat_title']
    # for outputs with pydantic
    return title.chat_title
