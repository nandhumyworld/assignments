import os
from dotenv import load_dotenv
import streamlit as st
import tempfile
import os
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

def get_openai_api_key():
    """
    Retrieves the OpenAI API key from environment variables.
    
    Returns:
        str: The OpenAI API key.
    """
       # Load environment variables from .env file
    load_dotenv(dotenv_path="./assignments/openai_p/.env")  # optional if .env is in the same dir
    openai_api_key = os.getenv("OPENAI_API_KEY") or "sk-your-api-key"
    
    # openai_api_key = st.sidebar.text_input(
    #     "OpenAI API Key",
    #     type="password",
    #     help="Enter your OpenAI API key to use embeddings"
    # )
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set it in the .env file.")
    
    return openai_api_key

def create_embeddings():
    """
    Create OpenAI embeddings instance
    
    Args:
        openai_api_key: OpenAI API key
        
    Returns:
        OpenAIEmbeddings instance
    """
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=get_openai_api_key(),
            model="text-embedding-ada-002"
        )
        return embeddings
        
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None
    
def get_OpenAI_client():
    """
    Get OpenAI client instance
    
    Returns:
        OpenAI client instance
    """
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=get_openai_api_key())
        return openai_client
        
    except Exception as e:
        st.error(f"Error getting OpenAI client: {str(e)}")
        return None