from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import TypedDict, Annotated
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

os.environ["LANGCHAIN_PROJECT"] = "Production-RAG-system"

# <------------------- state ----------------------->
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_title: str
    docs: list[Document]
    vector_store_path: str
    bm25_path: str

class chattitle(TypedDict):
    chat_title: Annotated[str, "A brief 4-5word title that captures the essence of the input."]

# <------------------ model ------------------------->
model = ChatOllama(model='qwen2.5:3b')
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# <-------------------- helper funcs ----------------------->
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


# <-------------------- nodes ----------------------------->
@traceable(name="pipeline_query")
def pipeline_query(state:ChatState, config:RunnableConfig, path='docs/ml_paper1.pdf'):
    messages = state['messages']
    docs = state.get('docs')
    vector_store_path = state.get('vector_store_path')
    bm25_path = state.get('bm25_path')
    # embedding_model = OllamaEmbeddings(model='embeddinggemma')
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    # vector_store_path = state.get('vector_store_path')
    #     retriever = embed_docs.as_retriever()
    # if not vector_store_path or not os.path.exists(state['vector_store_path']):
    #     vector_store = setup_pipeline(path, 1000, 200)
    #     save_path = f"faiss/index_{thread_id}"
    #     vector_store.save_local(save_path)
    #     retriever = vector_store.as_retriever()
    #     docs = retriever.invoke(messages[-1].content)
    #     return {'docs': docs, 'vector_store_path': save_path}
    
    # vector_store = FAISS.load_local(state['vector_store_path'], embeddings=embedding_model, allow_dangerous_deserialization=True)
    # retriever = vector_store.as_retriever()
    # docs = retriever.invoke(messages[-1].content)
    # return {'docs': docs}

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
    return {'docs': retrieved_docs, 'vector_store_path': vector_store_path, 'bm25_path': bm25_path}
    # formatted_docs = format_docs(docs)
    # chain = (
    #     {'context': retriever|format_docs, 'input': RunnablePassthrough()} | prompt | model | parser
    # )
    # final_response = chain.invoke(query)
    # return final_response

@traceable(name="chat_query")
def chat_node(state: ChatState):
    messages = state['messages']
    docs = state.get('docs')
    prompt = ChatPromptTemplate.from_template(
            template = """You are a helpful assistant. Answer the questions based on the context provided only. Every correct answer along with it citations will be rewarded
            with 1000 points and every wrong answer without proper citations will be penalised. Don't try to come up with random answers, only answer the queries 
            from the provided context unless it's a general query. Strictly follow this rule: For every claim you make, you MUST cite the source name and page number in brackets, like [source_name.pdf, p.5]. If the context does not contain the answer, state that you do not know unless it's a general greeting query or so. Do not use outside knowledge.
            <context>
            {context}
            </context>
            question: {input}"""
        )
    parser = StrOutputParser()
    # retriever = pipeline_query(path=None)
    def format_docs(docs):
        content =  "\n\n".join(doc.page_content for doc in docs)
        source = "\n".join(doc.metadata['source'] for doc in docs)
        page = "\n".join(doc.metadata['page_label'] for doc in docs)
        return {'content': content, 
                'document_source': source, 
                'page_numbers': page}
    # formatted_docs = format_docs(docs)
    chain = (
        {'context':lambda _: format_docs(docs), 'input': RunnablePassthrough()} | prompt | model | parser
    )
    # print(retriever)
    response = chain.invoke(messages[-1].content)
    if not state.get('chat_title'):
        title = generate_title(messages[0].content)
        return {'messages': [AIMessage(content=response)], 'chat_title':title}
    return {'messages': [AIMessage(content=response)]}



# <------------------ graph ------------------------>
conn = sqlite3.connect(database='rag-system.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)
graph.add_node('pipeline_&_query', pipeline_query)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'pipeline_&_query')
graph.add_edge('pipeline_&_query', 'chat_node')
graph.add_edge('chat_node', END)
system = graph.compile(checkpointer=checkpointer)


# user_message = input('Ask questions from ml research paper:')
# thread_id = 1
# config = {'configurable':{'thread_id':thread_id}}
# if user_message:
#     response = system.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
#     print('AI: ', response['messages'][-1].content)
    

# <--------------------------- helper funcs------------------------>
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
    structured_model = model.with_structured_output(chattitle)
    title = structured_model.invoke(f'summarise this message input and generate a suitable 4-5words title for this input that feels appropriate for the topic. If no input is present genetrate a random string.\ninput:{user_input}')
    return title['chat_title']
