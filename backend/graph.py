from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnablePassthrough
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
load_dotenv()
import os
from langsmith import traceable
from langchain_core.runnables import RunnableConfig
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import pickle
from pydantic import BaseModel, Field
import sys
from pathlib import Path
root_dir = Path(os.getcwd()).parent
sys.path.append(str(root_dir))
from backend.models import *
from backend.utils import *
from backend.checkpoint import checkpointer

os.environ["LANGCHAIN_PROJECT"] = "Production-RAG-system"

# <------------------- state ----------------------->
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_title: str
    docs: list[Document]
    vector_store_path: str
    bm25_path: str


# <-------------------- nodes ----------------------------->
@traceable(name="pipeline_query")
def pipeline_query(state:ChatState, config:RunnableConfig, path='docs/ml_paper1.pdf'):
    messages = state['messages']
    docs = state.get('docs')
    vector_store_path = state.get('vector_store_path')
    bm25_path = state.get('bm25_path')
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")

    if vector_store_path and bm25_path and os.path.exists(vector_store_path) and os.path.exists(bm25_path):
        vector_store = FAISS.load_local(state['vector_store_path'], embeddings=embedding_model, allow_dangerous_deserialization=True)
        with open(state['bm25_path'], "rb") as f:
            bm25_retriever = pickle.load(f)
    else: 
        split_docs = setup_pipeline(path, 1000, 200)
        vector_store = FAISS.from_documents(split_docs, embedding_model)
        vector_store_path = f"index/faiss_{thread_id}"
        vector_store.save_local(vector_store_path)
        bm25_retriever = BM25Retriever.from_documents(split_docs)
        bm25_path = f"index/bm25_{thread_id}"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
    
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k":5})
    bm25_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    retrieved_docs = ensemble_retriever.invoke(messages[-1].content)

    reranked_docs = rerank(retrieved_docs, messages[-1].content)
    return {'docs': retrieved_docs, 'vector_store_path': vector_store_path, 'bm25_path': bm25_path}


@traceable(name="chat_query")
def chat_node(state: ChatState):
    messages = state['messages']
    docs = state.get('docs')
    prompt = load_system_prompt()

    def format_docs_with_metadata(docs):
        formatted = []
        for i, doc in enumerate(docs):
            # Extract metadata safely
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page_label", "N/A")
            
            # Create a clearly delineated block
            doc_string = f"--- DOCUMENT ID: {i} | SOURCE: {source} | PAGE: {page} ---\n{doc.page_content}\n"
            formatted.append(doc_string)
    
        return "\n".join(formatted)
    # structured_model = model.with_structured_output(FinalResponse)
    chain = (
        {'context':lambda _: format_docs_with_metadata(docs), 'input': RunnablePassthrough()} | prompt | model
    )
    response = chain.invoke(messages[-1].content)
    if not state.get('chat_title'):
        title = generate_title(messages[0].content)
        return {'messages': [AIMessage(content=response.content)], 'chat_title':title}
    return {'messages': [AIMessage(content=response.content)]}



# <------------------ graph ------------------------>
graph = StateGraph(ChatState)
graph.add_node('pipeline_&_query', pipeline_query)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'pipeline_&_query')
graph.add_edge('pipeline_&_query', 'chat_node')
graph.add_edge('chat_node', END)
system = graph.compile()
