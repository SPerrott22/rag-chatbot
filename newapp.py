import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import ImageCaptionLoader
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain.callbacks.base import BaseCallbackHandler
import os
import pytube
import openai
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests

# class StreamHandler(BaseCallbackHandler):
#     def __init__(self, container, initial_text=""):
#         self.container = container
#         self.text = initial_text

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         self.text += token
#         self.container.markdown(self.text)


# Chat UI title
st.header("Upload your own context and ask questions like ChatGPT")
st.subheader('File types supported: PDF/DOCX/TXT/JPG/PNG/YouTube/Website :city_sunrise:')

# File uploader in the sidebar on the left
with st.sidebar:
    # Input for OpenAI API Key
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets['OPENAI_API_KEY']
    else:
        openai_api_key = st.text_input("OpenAI API Key", type="password")

    # Check if OpenAI API Key is provided
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Set OPENAI_API_KEY as an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize ChatOpenAI model
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", streaming=True) #max_tokens=16000


# Sidebar section for uploading files and providing a YouTube URL
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
    youtube_url = st.text_input("YouTube URL")
    website_url = st.text_input("Website URL")
    st.info("Please refresh the browser if you decide to upload more files to reset the session", icon="🚨")

# Check if files are uploaded or YouTube URL is provided
if uploaded_files or youtube_url or website_url:
    # Print the number of files uploaded or YouTube URL provided to the console
    st.write(f"Number of files uploaded: {len(uploaded_files)}")

    # Load the data and perform preprocessing only if it hasn't been loaded before
    if "processed_data" not in st.session_state:
        # Load the data from uploaded files
        documents = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Get the full file path of the uploaded file
                file_path = os.path.join(os.getcwd(), uploaded_file.name)

                # Save the uploaded file to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Check if the file is an image
                if file_path.endswith((".png", ".jpg")):
                    # Use ImageCaptionLoader to load the image file
                    image_loader = ImageCaptionLoader(path_images=[file_path])

                    # Load image captions
                    image_documents = image_loader.load()

                    # Append the Langchain documents to the documents list
                    documents.extend(image_documents)
                    
                elif file_path.endswith((".pdf", ".docx", ".txt")):
                    # Use UnstructuredFileLoader to load the PDF/DOCX/TXT file
                    loader = UnstructuredFileLoader(file_path)
                    loaded_documents = loader.load()

                    # Extend the main documents list with the loaded documents
                    documents.extend(loaded_documents)            
        
        # Load the YouTube audio stream if URL is provided
        if youtube_url:
            youtube_video = pytube.YouTube(youtube_url)
            streams = youtube_video.streams.filter(only_audio=True)
            stream = streams.first()
            stream.download(filename="youtube_audio.mp4")
            # Set the API key for Whisper
            openai.api_key = openai_api_key
            with open("youtube_audio.mp4", "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
            youtube_text = transcript['text']

            # Create a Langchain document instance for the transcribed text
            youtube_document = Document(page_content=youtube_text, metadata={})
            documents.append(youtube_document)
            
        # Load the website content if URL is provided
        if website_url:
            # def tag_visible(element):
            #     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            #         return False
            #     if isinstance(element, Comment):
            #         return False
            #     return True
            # webpage = requests.get(website_url)
            # html = BeautifulSoup(webpage.text, "html.parser")
            # texts = html.find_all(text=True)
            # visible_texts = filter(tag_visible, texts)  
            # # Use UnstructuredFileLoader to load the website content
            # loader = UnstructuredFileLoader()
            # loaded_documents = loader.load()
            # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            # text_splitter = SemanticChunker(OpenAIEmbeddings())

            # loaded_documents = text_splitter.create_documents([u" ".join(t.strip() for t in visible_texts)])
            loader = WebBaseLoader(website_url)
            loaded_documents = loader.load()
            # Extend the main documents list with the loaded documents
            documents.extend(loaded_documents)

        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

    else:
        # If the processed data is already available, retrieve it from session state
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

    # Initialize Langchain's QA Chain with the vectorstore
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), chain_type="stuff", memory=memory, max_tokens_limit=4000, return_source_documents=True, get_chat_history=lambda h : h)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your questions?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Query the assistant using the latest chat history
        history = [
            f"{message['role']}: {message['content']}" 
            for message in st.session_state.messages
        ]
    
        result = qa({
            "question": prompt, 
            "chat_history": history
        })
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)    
        print(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please upload your files and provide a YouTube URL for transcription.")