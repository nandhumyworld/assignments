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

def load_pdf_documents(uploaded_file) -> List[Document]:
    """
    Load PDF documents using LangChain's PyPDFLoader
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        List of Document objects
    """
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
        
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return []

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of split Document objects
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        return split_docs
        
    except Exception as e:
        st.error(f"Error splitting documents: {str(e)}")
        return []

def store_in_vectordb(documents: List[Document], embeddings, collection_name: str = "pdf_collection"):
    """
    Store documents in ChromaDB vector database
    
    Args:
        documents: List of Document objects
        embeddings: OpenAI embeddings instance
        collection_name: Name of the collection in ChromaDB
        
    Returns:
        ChromaDB vector store instance
    """
    try:
        # Create ChromaDB client with persistent storage
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create vector store
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        # Add documents to vector store
        vectorstore.add_documents(documents)
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error storing in vector database: {str(e)}")
        return None
      
def get_retriever(vectorstore, k: int = 5):
    """
    Get retriever from vector store
    
    Args:
        vectorstore: ChromaDB vector store instance
        k: Number of documents to retrieve
        
    Returns:
        Retriever instance
    """
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        return retriever
        
    except Exception as e:
        st.error(f"Error getting retriever: {str(e)}")
        return None

def retrieve_from_vectordb(vectorstore, query: str, k: int = 5) -> List[Document]:
    """
    Retrieve relevant documents from vector database
    
    Args:
        vectorstore: ChromaDB vector store instance
        query: Search query
        k: Number of documents to retrieve
        
    Returns:
        List of relevant Document objects
    """
    try:
        # Perform similarity search
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
#         # Perform similarity search with score threshold
#         retriever = vectorstore.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={
#         "k": 5,
#         "score_threshold": 0.7  # only return results with score ≥ 0.7
#         }
# )

        
        relevant_docs = retriever.get_relevant_documents(query)
        return relevant_docs
        
    except Exception as e:
        st.error(f"Error retrieving from vector database: {str(e)}")
        return []