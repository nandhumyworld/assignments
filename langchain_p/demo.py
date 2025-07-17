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
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Set page config
st.set_page_config(
    page_title="LangChain RAG Pipeline Demo",
    page_icon="📚",
    layout="wide"
)

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

def create_embeddings(openai_api_key: str):
    """
    Create OpenAI embeddings instance
    
    Args:
        openai_api_key: OpenAI API key
        
    Returns:
        OpenAIEmbeddings instance
    """
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-ada-002"
        )
        return embeddings
        
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

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

def main():
    st.title("📚 LangChain RAG Pipeline Demo")
    st.markdown("Upload a PDF document and ask questions about its content using vector similarity search!")
    
    # Sidebar for API key
    st.sidebar.header("Configuration")

    # Load environment variables from .env file
    load_dotenv(dotenv_path="./assignments/openai_p/.env")  # optional if .env is in the same dir
    openai_api_key = os.getenv("OPENAI_API_KEY") or "sk-your-api-key"
    
    # openai_api_key = st.sidebar.text_input(
    #     "OpenAI API Key",
    #     type="password",
    #     help="Enter your OpenAI API key to use embeddings"
    # )
    
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # File upload section
    st.header("1. Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to process"
    )
    
    if uploaded_file is not None and openai_api_key:
        st.success(f"PDF uploaded: {uploaded_file.name}")
        
        # Process button
        if st.button("Process PDF", type="primary"):
            if not openai_api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
                return
                
            with st.spinner("Processing PDF..."):
                # Step 1: Load PDF
                st.write("📄 Loading PDF documents...")
                documents = load_pdf_documents(uploaded_file)
                
                if documents:
                    st.success(f"Loaded {len(documents)} pages from PDF")
                    
                    # Step 2: Split documents
                    st.write("✂️ Splitting documents into chunks...")
                    split_docs = split_documents(documents)
                    st.success(f"Split into {len(split_docs)} chunks")
                    
                    # Step 3: Create embeddings
                    st.write("🔢 Creating embeddings...")
                    embeddings = create_embeddings(openai_api_key)
                    
                    if embeddings:
                        st.success("Embeddings created successfully")
                        
                        # Step 4: Store in vector database
                        st.write("🗄️ Storing in vector database...")
                        vectorstore = store_in_vectordb(split_docs, embeddings)
                        
                        if vectorstore:
                            st.success("Documents stored in vector database")
                            st.session_state.vectorstore = vectorstore
                            st.session_state.processed = True
                        else:
                            st.error("Failed to store documents in vector database")
                    else:
                        st.error("Failed to create embeddings")
                else:
                    st.error("Failed to load PDF documents")
    
    # Query section
    if st.session_state.processed and st.session_state.vectorstore:
        st.header("2. Ask Questions")
        
        # Query input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?",
                key="query_input"
            )
        
        with col2:
            st.write("")  # Empty space for alignment
            ask_button = st.button("Ask from VectorDB", type="primary")
        
        with col2:
            st.write("")  # Empty space for alignment
            ask_llm_button = st.button("Ask from LLM", type="primary")
    
        # Process query
        if ask_button and query:
            with st.spinner("Retrieving relevant information..."):
                # Step 5: Retrieve from vector database
                relevant_docs = retrieve_from_vectordb(
                    st.session_state.vectorstore, 
                    query, 
                    k=5 
                )
                
                if relevant_docs:
                    st.header("3. Retrieved Results")
                    st.write(f"Found {len(relevant_docs)} relevant document chunks:")
                    
                    # Display results
                    for i, doc in enumerate(relevant_docs, 1):
                        with st.expander(f"Result {i}", expanded=True):
                            st.write("**Content:**")
                            st.write(doc.page_content)
                            
                            if doc.metadata:
                                st.write("**Metadata:**")
                                st.json(doc.metadata)
                            
                            st.write("---")
                else:
                    st.warning("No relevant documents found for your query.")
        elif ask_llm_button and query:
            with st.spinner("Retrieving relevant information..."):
            # Step 2: Build a RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(),
                retriever=get_retriever(
                    st.session_state.vectorstore, 
                    k=5),
                    return_source_documents=False,
                chain_type="stuff",)  # could be 'stuff', 'map_reduce', or 'refine
                answer = qa_chain.run(query)
                st.write("### Answer from LLM:")
                st.write(answer)
        
        elif ask_button and not query:
            st.warning("Please enter a question to search for.")
    
    elif not openai_api_key:
        st.info("Please enter your OpenAI API key in the sidebar to get started.")
    
    elif not uploaded_file:
        st.info("Please upload a PDF document to begin processing.")
    
    # Information section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This application demonstrates a complete LangChain RAG (Retrieval-Augmented Generation) pipeline:
        
        1. **PDF Loading**: Uses `PyPDFLoader` to extract text from uploaded PDF documents
        2. **Document Splitting**: Uses `RecursiveCharacterTextSplitter` to break documents into manageable chunks
        3. **Embeddings**: Uses OpenAI's `text-embedding-ada-002` model to create vector representations
        4. **Vector Storage**: Stores embeddings in ChromaDB for efficient similarity search
        5. **Retrieval**: Searches for relevant document chunks based on query similarity
        
        **Requirements:**
        - OpenAI API key for embeddings
        - PDF document to process
        - Internet connection for API calls
        """)
    
    # Technical details
    with st.expander("🔧 Technical Configuration"):
        st.markdown("""
        **Text Splitter Settings:**
        - Chunk size: 1000 characters
        - Chunk overlap: 200 characters
        - Separators: Paragraphs, lines, spaces
        
        **Embedding Model:**
        - OpenAI text-embedding-ada-002
        - Dimension: 1536
        
        **Vector Database:**
        - ChromaDB with persistent storage
        - Similarity search with cosine distance
        - Default retrieval: Top 5 results
        """)

if __name__ == "__main__":
    main()