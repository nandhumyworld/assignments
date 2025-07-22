# LangChain RAG Pipeline Demo

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Streamlit, and OpenAI embeddings. Users can upload PDF documents, process them into vector databases, and ask questions about their content using similarity search and advanced RAG techniques.

## Features
- Upload and process PDF documents
- Split documents into manageable chunks
- Generate embeddings using OpenAI
- Store and retrieve document chunks using ChromaDB
- Ask questions and get relevant answers from the document
- Self RAG and Corrective RAG options for improved retrieval
- Interactive Streamlit UI

## Usage
1. Install dependencies from `requirements.txt` (see below)
2. Run the app:
   ```bash
   streamlit run main.py
   ```
3. Enter your OpenAI API key when prompted
4. Upload a PDF and start asking questions!

## File Structure
- `main.py`: Main Streamlit app
- `openai_util.py`: OpenAI API utilities
- `langchain_fw.py`: LangChain document and vector DB functions
- `self_rag.py`: Self RAG logic
- `corrective_rag.py`: Corrective RAG logic

## Requirements
See `requirements.txt` for all required packages.

## License
MIT
