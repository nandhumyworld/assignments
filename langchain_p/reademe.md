I'll create a comprehensive Streamlit application that demonstrates the LangChain RAG (Retrieval-Augmented Generation) pipeline with all the components you requested. This will include PDF loading, document splitting, embedding, vector storage, and retrieval.I've created a comprehensive Streamlit application that demonstrates the complete LangChain RAG pipeline. Here's what the application includes:

## Key Features:

### 1. **Separate Functions for Each Component:**

- `load_pdf_documents()` - Uses PyPDFLoader to extract text from PDFs
- `split_documents()` - Uses RecursiveCharacterTextSplitter for chunking
- `create_embeddings()` - Creates OpenAI embeddings instance
- `store_in_vectordb()` - Stores documents in ChromaDB
- `retrieve_from_vectordb()` - Retrieves relevant documents using similarity search
- `retrieve_from_openai()` - Retrieves results using open ai API call

### 2. **Complete Streamlit UI:**

- PDF upload functionality
- API key input in sidebar
- Process button to run the pipeline
- Query input with Ask button
- Results display in expandable sections

### 3. **Technical Implementation:**

- Uses OpenAI's `text-embedding-ada-002` model for embeddings
- ChromaDB for persistent vector storage
- Configurable chunk size (1000 chars) and overlap (200 chars)
- Similarity search with top-k retrieval (default: 5 results)

# Changes to be done in below code

Refer How_to_configure_API_key.jpg in GIT repo

## To Run This Application:

1. **Install required packages:**

```bash
pip install streamlit langchain openai chromadb pypdf

```

1. **Run the application:**

```bash
streamlit run demo.py

```

1. **Usage:**
    - Enter your OpenAI API key in the sidebar
    - Upload a PDF document
    - Click "Process PDF" to run the pipeline
    - Ask questions in the text box and click "Ask"
    - View retrieved relevant chunks below

## Key LangChain Concepts Demonstrated:

- **Document Loading**: PyPDFLoader for PDF processing
- **Text Splitting**: RecursiveCharacterTextSplitter for intelligent chunking
- **Embeddings**: OpenAI embeddings for vector representation
- **Vector Storage**: ChromaDB for persistent storage
- **Retrieval**: Similarity search for relevant document chunks

The application provides a complete RAG pipeline with proper error handling, progress indicators, and detailed information about how each component works. Each function is modular and can be easily modified or extended for different use cases.