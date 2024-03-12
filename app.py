import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
import tiktoken
from langchain.text_splitter import CharacterTextSplitter
# from openai import OpenAI
import streamlit as st



# import _thread
# import ssl
# import threading

# import httpcore


# # Define a custom hash function for _thread.RLock objects
# def hash_rlock(rlock):
#     # Return a constant because there's no meaningful internal state to hash
#     return 0


# # Define a custom hash function for _thread.lock objects
# def hash_thread_lock(lock):
#     # Since lock objects generally manage access to resources and their state
#     # is external to the function's deterministic output, you might choose to
#     # return a constant. However, ensure this approach fits your application's logic.
#     return 42

# # Define a custom hash function for ssl.SSLContext objects
# def hash_ssl_context(ctx):
#     # Return a constant because the SSL context's details don't affect the function's output
#     return 1

# # Define a custom hash function for ssl.SSLSocket objects
# def hash_sslsocket(socket):
#     # Return a constant to "fake" hashing the socket
#     # WARNING: This approach has significant implications for caching correctness!
#     return 42


# # Use the custom hash function in the st.cache decorator
# @st.cache(hash_funcs={_thread.RLock: hash_rlock, ssl.SSLSocket: hash_sslsocket, ssl.SSLContext: hash_ssl_context, threading.Lock: hash_thread_lock, httpcore._synchronization.ThreadLock: hash_thread_lock})
def create_vectorstore():
    # Process each file
    vectorstore = None    
    for txt_file in txt_files:
        file_path = os.path.join('./', txt_file)
        
        # Assuming you have a TextLoader and it works like this
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        
        # Assuming you have a CharacterTextSplitter and it works like this
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_documents = text_splitter.split_documents(documents)
        
        for document in splitted_documents:
            if vectorstore is None:
                # Assuming this is how you initialize your FAISS vectorstore and conversational chain
                vectorstore = FAISS.from_documents([document], embeddings)
                # chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
                # chat_history = []
            else:
                vectorstore.add_documents([document])
            time.sleep(2)  # Add delay here
    return vectorstore

def create_llm(vectorstore):
    memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=memory
            )
    return conversation_chain
# You might need to install or adjust imports depending on your environment

# Create a vector store


# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def response_generator():
    # response = random.choice(
    #     [
    #         "Hello there! How can I assist you today?",
    #         "Hi, human! Is there anything I can help you with?",
    #         "Do you need help?",
    #     ]
    # )
    result = conversation_chain({"question": prompt})
    response = result["answer"]

    for word in response.split():
        yield word + " "
        time.sleep(0.05)


llm_model = "gpt-3.5-turbo"
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.7, model_name=llm_model)

txt_files = [f for f in os.listdir('./') if f.endswith('.txt')]

vector_store = create_vectorstore()

conversation_chain = create_llm(vector_store)

# Set page config
st.set_page_config(page_title="DSU RAG Chatbot", page_icon="📊")

st.title("DSU RAG Chatbot")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # stream = llm.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        response = st.write_stream(response_generator())
        # response = st.write_stream(answer)
    st.session_state.messages.append({"role": "assistant", "content": response})



# st.title("DSU RAG Chatbot")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    

# # Display assistant response in chat message container
# with st.chat_message("assistant"):
#     response = st.write_stream(response_generator())
# # Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "content": response})



# Title
# st.title("DSU Data Processing App")

# Prompt user for API key and other configurations
# api_key = st.text_input("Enter your OpenAI API key:", type="password")
# if api_key:
#     os.environ["OPENAI_API_KEY"] = api_key
#     st.success("API Key Set!")

# # Section to display team members
# st.subheader("Team Members")
# if st.button("Fetch Team Members"):
#     webpage = requests.get("https://datascienceunion.com/team")
#     soup = BeautifulSoup(webpage.text, "html.parser")
#     texts = soup.find_all('ul')  # Modify this as needed

#     extracted_text = '\n'.join([text.get_text() for text in texts])
#     items = [item for item in extracted_text.split('\n') if item.strip()]

#     members = ""
#     for i in range(0, len(items), 2):
#         name = items[i].strip()
#         title_or_class = items[i+1].strip() if (i + 1) < len(items) else ""
#         members += f"{name} - {title_or_class}\n"

#     st.text(members)

# # Section for processing documents
st.subheader("Process Documents")
# # Placeholder for document processing logic
# # Due to complexity, actual LangChain integration is not shown

# # Optionally, you can add file uploaders for users to upload documents
# # and buttons to trigger various actions based on uploaded content.

# # Example: Upload a document
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Process the file here
    # loader = TextLoader(uploaded_file, encoding="utf-8")
    # documents = loader.load()
    
    # Assuming you have a CharacterTextSplitter and it works like this
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_documents = text_splitter.split_documents(uploaded_file) # documents
    
    for document in splitted_documents:
        if vector_store is None:
            # Assuming this is how you initialize your FAISS vectorstore and conversational chain
            vector_store = FAISS.from_documents([document], embeddings)
            # chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
            # chat_history = []
        else:
            vector_store.add_documents([document])
        time.sleep(2)  # Add delay here

    st.write("File uploaded successfully!")

# This is a basic structure. Depending on your requirements, you may need to add more functionality,
# adjust how you handle file paths, and interact with APIs or external services.
