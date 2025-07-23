import streamlit as st
import openai_util as llm
import langchain_fw as lc
import self_rag as sRag
from corrective_rag import CorrectiveRAG
import corrective_rag as cRag


# Set page config
st.set_page_config(
    page_title="LangChain RAG Pipeline Demo",
    page_icon="📚",
    layout="wide"
)

def main():
    st.title("📚 LangChain RAG Pipeline Demo")
    st.markdown("Upload a PDF document and ask questions about its content using vector similarity search!")

    openai_api_key = llm.get_openai_api_key()

    
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
                documents = lc.load_pdf_documents(uploaded_file)
                
                if documents:
                    st.success(f"Loaded {len(documents)} pages from PDF")
                    
                    # Step 2: Split documents
                    st.write("✂️ Splitting documents into chunks...")
                    split_docs = lc.split_documents(documents)
                    st.success(f"Split into {len(split_docs)} chunks")
                    
                    # Step 3: Create embeddings
                    st.write("🔢 Creating embeddings...")
                    embeddings = llm.create_embeddings()
                    
                    if embeddings:
                        st.success("Embeddings created successfully")
                        
                        # Step 4: Store in vector database
                        st.write("🗄️ Storing in vector database...")
                        vectorstore = lc.store_in_vectordb(split_docs, embeddings)
                        
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
            user_query = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?",
                key="query_input"
            )

            # user_query = "What is my designation that they offer me ?"  # Example query for testing
            # st.write(f"**Example Query:** {user_query}")
        
        with col2:
            st.write("")  # Empty space for alignment
            ask_button = st.button("Ask from VectorDB", type="primary")
        
        with col2:
            st.write("")  # Empty space for alignment
            self_rag_button = st.button("Use self RAG", type="primary")
        with col2:
            st.write("")  # Empty space for alignment
            corrective_rag_button = st.button("Use corrective RAG", type="primary")
    
        # Process query
        if ask_button and user_query:
            with st.spinner("Retrieving relevant information..."):
                # Step 5: Retrieve from vector database
                relevant_docs = lc.retrieve_from_vectordb(
                    st.session_state.vectorstore, 
                    user_query, 
                    k=1
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
        elif self_rag_button and user_query:
            with st.spinner("Retrieving relevant information..."):
                retrieved_context = sRag.self_rag(user_query, lc.get_retriever(
                                            st.session_state.vectorstore, 
                                            k=1))
                st.write("### Answer from LLM using Self RAG:")
                st.write(retrieved_context)
        elif corrective_rag_button and user_query:
            with st.spinner("Retrieving relevant information..."):
                retrieved_context = sRag.self_rag(user_query, lc.get_retriever(
                                            st.session_state.vectorstore, 
                                            k=1))                

                # Initialize the corrective RAG system
                corrective_rag_instance = CorrectiveRAG(st.session_state)

                evaluation = corrective_rag_instance.context_evaluation(user_query, retrieved_context, st.session_state)

                # Make correction decision
                decision = corrective_rag_instance.correction_decision(evaluation, user_query)

                # Format and display response
                if decision.action.value.strip() == cRag.ActionType.PROCEED_WITH_ANSWER.value:
                    response = corrective_rag_instance.format_response(
                        user_query, retrieved_context, evaluation, decision,
                        answer=retrieved_context,
                        sources=[uploaded_file.name]
                    )
                    st.write("### Answer from LLM  using Corrective RAG:")
                    st.write(response)
                else:
                    # Use decision.new_query to retrieve again
                    print(f"Need to retrieve again with query: {decision.new_query}")
        
        elif ask_button and not user_query:
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