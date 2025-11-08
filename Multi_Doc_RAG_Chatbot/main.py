import os
import json

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from vectorize_documents import embeddings

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore


def chat_chain(vectorstore):
    # Updated to current Groq model
    llm = ChatGroq(model="llama-3.3-70b-versatile",
                   temperature=0)
    retriever = vectorstore.as_retriever()
    
    # FIX: Added output_key to specify which output to store in memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Explicitly tell memory to store the 'answer' key
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=True,
        output_key="answer"  # Ensure consistent output key
    )

    return chain


st.set_page_config(
    page_title="Multi Doc Chat",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Multi Documents Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = st.session_state.conversational_chain({"question": user_input})
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            
            # Optional: Show source documents (uncomment if you want to see sources)
            # with st.expander("Source Documents"):
            #     for i, doc in enumerate(response["source_documents"]):
            #         st.write(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page', 'N/A')}")
            #         st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})