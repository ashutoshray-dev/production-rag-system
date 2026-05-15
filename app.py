import streamlit as st
from backend.graph import system
from backend.utils import retrieve_threads_list
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
import uuid

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_history(thread_id):
    if 'messages' not in system.get_state(config={'configurable':{'thread_id':thread_id}}).values:
        return []
    else:
        return system.get_state(config={'configurable':{'thread_id':thread_id}}).values['messages']
def load_chat(thread_id):
    if 'chat_title' not in system.get_state(config={'configurable':{'thread_id':thread_id}}).values:
        return " "
    else:
        return system.get_state(config={'configurable':{'thread_id':thread_id}}).values['chat_title']

# uploaded_file = st.file_uploader(
#     "Upload file", accept_multiple_files=False, type="pdf"
# )

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_threads_list()

add_thread(st.session_state['thread_id'])

st.sidebar.title('rag-research-system')
if st.sidebar.button('New Chat'):
    reset_chat()
st.sidebar.header('My conversations')
for thread_id in st.session_state['chat_threads'][::-1]:
    title = load_chat(thread_id)
    if st.sidebar.button(label=str(title), key=thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_history(thread_id=thread_id)
        temp_message = []
        for msg in messages:
            # print(f"DEBUG: msg type is {type(msg)} and content is {msg.content[:20]}")
            if isinstance(msg, HumanMessage):
                temp_message.append({'role':'user', 'content':msg.content})
            elif isinstance(msg, AIMessage):
                temp_message.append({'role':'assistant', 'content':msg.content})
        st.session_state['message_history'] = temp_message

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

CONFIG = {'configurable':{'thread_id':st.session_state['thread_id']}}
user_input = st.chat_input('Enter here')
if user_input:
    st.session_state['message_history'].append({'role':'user', 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        def ai_message_stream():
            for message_chunk in system.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages',
                version='v2'
            ):
                if isinstance(message_chunk['data'][0], AIMessageChunk):
                    yield message_chunk['data'][0].content
        ai_message = st.write_stream(ai_message_stream())
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})