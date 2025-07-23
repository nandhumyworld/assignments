import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import csv
from typing import List, Dict
import faiss
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from openai_util import *
import warnings

warnings.filterwarnings("ignore")

# Configuration
OPENAI_API_KEY = get_openai_api_key()

# File paths
CSV_FILE = "journal_entries.csv"
FAISS_INDEX_FILE = "journal_faiss.index"
FAISS_METADATA_FILE = "journal_metadata.pkl"

class JournalingApp:
    def __init__(self):
        self.csv_file = CSV_FILE
        self.faiss_index_file = FAISS_INDEX_FILE
        self.faiss_metadata_file = FAISS_METADATA_FILE
           
        # Initialize embeddings and LLM            
        if OPENAI_API_KEY:
            self.embeddings = create_embeddings()
            self.llm = get_ChatOpenAI_client()
            
            # Get embedding dimension (OpenAI embeddings are typically 1536)
            try:
                test_embedding = self.embeddings.embed_query("test")
                self.embedding_dim = len(test_embedding)
                st.info(f"📏 Detected embedding dimension: {self.embedding_dim}")
            except Exception as e:
                st.error(f"Error detecting embedding dimension: {e}")
                self.embedding_dim = 1536  # Default for OpenAI
        else:
            st.error("OpenAI API Key not found. Please set OPENAI_API_KEY in secrets.")
            return
            
        # Initialize CSV file
        self.init_csv()
        
        # Load or create FAISS index
        self.load_faiss_index()
        
        # Load existing CSV data into vector database if FAISS is empty
        if self.index.ntotal == 0:
            self.load_csv_to_vector_db()
        
        # Journal questions
        self.journal_questions = {
            "Reflect briefly": [
                "What do I want from today?",
                "What am I grateful for?",
                "What will I do if things go wrong?"
            ],
            "Design your day": [
                "What must get done today?",
                "What would make today feel meaningful?",
                "What could derail me—and how will I respond?"
            ],
            "Wrap-Up": [
                "What did you learn today?",
                "What did you contribute today?",
                "What's still on my mind?",
                "What needs to be carried over to tomorrow?",
                "One word that captures today is:"
            ]
        }
    
    def init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'date', 'time', 'mode', 'question', 'answer', 'entry_id'
                ])
    
    def load_faiss_index(self):
        """Load existing FAISS index or create new one"""
        try:
            if os.path.exists(self.faiss_index_file) and os.path.exists(self.faiss_metadata_file):
                self.index = faiss.read_index(self.faiss_index_file)
                with open(self.faiss_metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                st.success(f"✅ Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index with correct dimensions
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.metadata = []
                st.info(f"📝 Created new FAISS index ({self.embedding_dim} dimensions)")
                # Save empty index immediately
                self.save_faiss_index()
        except Exception as e:
            st.error(f"❌ Error loading FAISS index: {e}")
            st.exception(e)
            # Create new index as fallback
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            self.save_faiss_index()
    
    def save_faiss_index(self):
        """Save FAISS index to disk"""
        try:
            # Ensure directory exists
            index_dir = os.path.dirname(self.faiss_index_file)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
            
            faiss.write_index(self.index, self.faiss_index_file)
            with open(self.faiss_metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            st.success(f"✅ FAISS index saved successfully! Total vectors: {self.index.ntotal}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving FAISS index: {e}")
            st.exception(e)
            return False
    
    def load_csv_to_vector_db(self):
        """Load all existing CSV data into vector database"""
        if not os.path.exists(self.csv_file):
            return
            
        try:
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return
                
            st.info("🔄 Loading existing CSV data into vector database...")
            
            # Group by entry_id to combine related Q&As
            grouped = df.groupby('entry_id')
            
            added_count = 0
            for entry_id, group in grouped:
                # Combine all Q&As for this entry
                combined_text = f"Date: {group.iloc[0]['date']}, Mode: {group.iloc[0]['mode']}\n"
                answers_dict = {}
                
                for _, row in group.iterrows():
                    combined_text += f"Q: {row['question']}\nA: {row['answer']}\n"
                    answers_dict[row['question']] = row['answer']
                
                # Create metadata
                metadata = {
                    'date': group.iloc[0]['date'],
                    'mode': group.iloc[0]['mode'],
                    'questions': list(answers_dict.keys()),
                    'answers': answers_dict,
                    'entry_id': entry_id
                }
                
                # Add to vector database
                if self.add_to_vector_db_internal(combined_text, metadata):
                    added_count += 1
            
            if added_count > 0:
                st.success(f"✅ Successfully loaded {added_count} entries from CSV into vector database!")
                
        except Exception as e:
            st.error(f"❌ Error loading CSV to vector DB: {e}")
            st.exception(e)
    
    def save_to_csv(self, date_str: str, mode: str, question: str, answer: str, entry_id: str):
        """Save journal entry to CSV"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([date_str, timestamp, mode, question, answer, entry_id])
    
    def add_to_vector_db_internal(self, text: str, metadata: Dict) -> bool:
        """Internal method to add text to FAISS vector database without UI updates"""
        try:
            if not hasattr(self, 'embeddings'):
                return False
                
            embedding = self.embeddings.embed_query(text)
            embedding_array = np.array([embedding], dtype=np.float32)
            
            self.index.add(embedding_array)
            self.metadata.append(metadata)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding to vector DB: {e}")
            return False
    
    def add_to_vector_db(self, text: str, metadata: Dict) -> bool:
        """Add text to FAISS vector database with UI updates"""
        try:
            if not hasattr(self, 'embeddings'):
                st.error("❌ Embeddings not initialized. Cannot add to vector DB.")
                return False
                
            st.info("🔄 Generating embedding...")
            embedding = self.embeddings.embed_query(text)
            embedding_array = np.array([embedding], dtype=np.float32)
            
            st.info("🔄 Adding to FAISS index...")
            self.index.add(embedding_array)
            self.metadata.append(metadata)
            
            st.info("🔄 Saving FAISS index to disk...")
            success = self.save_faiss_index()
            
            if success:
                st.success(f"✅ Successfully added entry to vector database! Total entries: {self.index.ntotal}")
            
            return success
            
        except Exception as e:
            st.error(f"❌ Error adding to vector DB: {e}")
            st.exception(e)
            return False
    
    def search_vector_db(self, query: str, k: int = 5) -> List[Dict]:
        """Search FAISS vector database"""
        try:
            if not hasattr(self, 'embeddings') or self.index.ntotal == 0:
                return []
                
            query_embedding = self.embeddings.embed_query(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            
            distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    results.append({
                        'metadata': self.metadata[idx],
                        'distance': distances[0][i]
                    })
            return results
        except Exception as e:
            st.error(f"Error searching vector DB: {e}")
            return []
    
    def generate_rag_response(self, query: str) -> str:
        """Generate response using RAG with FAISS search"""
        try:
            # Search for relevant journal entries
            search_results = self.search_vector_db(query, k=5)
            
            if not search_results:
                return "I don't have any relevant journal entries to answer your question. Please add some journal entries first."
            
            # Prepare context from search results
            context = ""
            for result in search_results:
                metadata = result['metadata']
                context += f"Date: {metadata['date']}, Mode: {metadata['mode']}\n"
                for q, a in metadata['answers'].items():
                    context += f"Q: {q}\nA: {a}\n"
                context += "\n"
            
            # Generate response using OpenAI
            system_message = SystemMessage(content="""
            You are a helpful journaling assistant. Based on the user's previous journal entries provided as context, 
            answer their question thoughtfully and personally. Reference specific entries when relevant.
            Keep your response supportive and insightful.
            """)
            
            human_message = HumanMessage(content=f"""
            Based on my journal entries below, please answer my question: {query}
            
            Journal Context:
            {context}
            """)
            
            response = self.llm([system_message, human_message])
            return response.content
            
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    st.set_page_config(
        page_title="Daily Journaling App",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 My Personal Daily Journaling App")
    st.markdown("---")
    
    # Initialize the app
    if 'journal_app' not in st.session_state:
        st.session_state.journal_app = JournalingApp()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    
    if 'current_questions' not in st.session_state:
        st.session_state.current_questions = []
    
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
    
    if 'current_answers' not in st.session_state:
        st.session_state.current_answers = {}
    
    app = st.session_state.journal_app

    def save_current_session_to_vectorstore():
        """Save current session answers to vector database"""
        if not st.session_state.current_answers:
            st.warning("No answers to save. Please complete some questions first.")
            return False
            
        selected_date = st.session_state.get('journal_date', date.today())
        selected_mode = st.session_state.get('journal_mode', 'Reflect briefly')
        
        # Combine all answers for embedding
        combined_text = f"Date: {selected_date.strftime('%Y-%m-%d')}, Mode: {selected_mode}\n"
        for q, a in st.session_state.current_answers.items():
            combined_text += f"Q: {q}\nA: {a}\n"
                            
        # Add to vector database
        metadata = {
            'date': selected_date.strftime('%Y-%m-%d'),
            'mode': selected_mode,
            'questions': list(st.session_state.current_answers.keys()),
            'answers': st.session_state.current_answers,
            'entry_id': f"{selected_date.strftime('%Y%m%d')}_{selected_mode.replace(' ', '_')}"
        }
                            
        with st.spinner("Creating embeddings and saving to vector database..."):
            success = app.add_to_vector_db(combined_text, metadata)
        
        if success:
            # Completion message
            completion_msg = f"✅ Completed {selected_mode} session! Your entries have been saved and indexed for future reference."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": completion_msg
            })
            return True
        return False
    
    # Sidebar for date and mode selection
    with st.sidebar:
        st.header("📅 Journal Settings")
        
        # Date selection
        selected_date = st.date_input(
            "Select Date",
            value=date.today(),
            key="journal_date"
        )
        
        # Mode selection
        modes = ["Reflect briefly", "Design your day", "Wrap-Up", "Ask"]
        selected_mode = st.radio(
            "Select Mode",
            modes,
            key="journal_mode"
        )

        # Save button - only show if there are answers to save
        if st.session_state.current_answers and selected_mode != "Ask":
            if st.button("💾 Save Current Session to Vector DB"):
                if save_current_session_to_vectorstore():
                    st.rerun()

        # Rebuild vector DB from CSV button
        if st.button("🔄 Rebuild Vector DB from CSV"):
            if hasattr(app, 'index'):
                # Clear existing index
                app.index = faiss.IndexFlatL2(app.embedding_dim)
                app.metadata = []
                
                # Reload from CSV
                app.load_csv_to_vector_db()
                st.rerun()

        # Reset session button
        if st.button("🆕 Start New Session"):
            st.session_state.chat_history = []
            st.session_state.current_mode = None
            st.session_state.current_questions = []
            st.session_state.question_index = 0
            st.session_state.current_answers = {}
            st.rerun()
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"Mode: {selected_mode}")
        st.subheader(f"Date: {selected_date.strftime('%Y-%m-%d')}")
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "assistant":
                    st.markdown(f"🤖 **Assistant:** {message['content']}")
                else:
                    st.markdown(f"👤 **You:** {message['content']}")
        
        # Handle different modes
        if selected_mode == "Ask":
            # RAG Chat mode
            st.markdown("---")
            st.subheader("💬 Ask about your journal entries")
            
            user_query = st.text_input("Ask me anything about your journal entries:", key="rag_query")
            
            if st.button("Send", key="send_rag"):
                if user_query.strip():
                    # Add user message to chat
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_query
                    })
                    
                    # Generate RAG response
                    with st.spinner("Searching through your journal entries..."):
                        response = app.generate_rag_response(user_query)
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()
        
        else:
            # Journal entry modes
            if st.session_state.current_mode != selected_mode:
                # Starting new mode
                st.session_state.current_mode = selected_mode
                st.session_state.current_questions = app.journal_questions[selected_mode]
                st.session_state.question_index = 0
                st.session_state.current_answers = {}
                
                # Add welcome message
                welcome_msg = f"Welcome to {selected_mode} mode! Let's begin with your journal entry for {selected_date.strftime('%Y-%m-%d')}."
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": welcome_msg
                })
            
            # Current journaling session
            if st.session_state.question_index < len(st.session_state.current_questions):
                current_question = st.session_state.current_questions[st.session_state.question_index]
                
                st.markdown("---")
                st.subheader(f"Question {st.session_state.question_index + 1} of {len(st.session_state.current_questions)}")
                st.markdown(f"**{current_question}**")
                
                user_answer = st.text_area("Your answer:", key=f"answer_{st.session_state.question_index}")
                
                if st.button("Submit Answer", key=f"submit_{st.session_state.question_index}"):
                    if user_answer.strip():
                        # Save answer
                        st.session_state.current_answers[current_question] = user_answer
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": current_question
                        })
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_answer
                        })
                        
                        # Save to CSV
                        entry_id = f"{selected_date.strftime('%Y%m%d')}_{selected_mode.replace(' ', '_')}_{st.session_state.question_index}"
                        app.save_to_csv(
                            selected_date.strftime('%Y-%m-%d'),
                            selected_mode,
                            current_question,
                            user_answer,
                            entry_id
                        )
                        
                        # Move to next question
                        st.session_state.question_index += 1
                        
                        # If all questions answered, auto-save to vector store
                        if st.session_state.question_index >= len(st.session_state.current_questions):
                            if save_current_session_to_vectorstore():
                                # Clear current session after successful save
                                st.session_state.current_answers = {}
                                st.session_state.question_index = 0
                        
                        st.rerun()
                    else:
                        st.warning("Please provide an answer before submitting.")
            
            else:
                st.success(f"✅ You have completed all questions for {selected_mode} mode!")
                st.info("You can start a new session or switch to Ask mode to query your journal entries.")
    
    with col2:
        st.header("📊 Journal Stats")
        
        # Load and display stats
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            
            st.metric("📝 CSV Entries", len(df))
            st.metric("📅 Days Journaled", df['date'].nunique())
            st.metric("🔍 Vector DB Entries", app.index.ntotal if hasattr(app, 'index') else 0)
            
            # Recent entries
            st.subheader("Recent Entries")
            if not df.empty:
                recent_entries = df.tail(5)[['date', 'mode', 'question']].to_dict('records')
                for entry in reversed(recent_entries):
                    st.text(f"{entry['date']} - {entry['mode']}")
        else:
            st.info("No journal entries yet. Start journaling!")
        
        # Current session info
        if st.session_state.current_answers:
            st.subheader("Current Session")
            st.info(f"Answered: {len(st.session_state.current_answers)} questions")
            for q, a in st.session_state.current_answers.items():
                with st.expander(q[:50] + "..."):
                    st.write(a)
        
        # Export options
        st.subheader("📤 Export Data")
        if st.button("Download CSV"):
            if os.path.exists(CSV_FILE):
                with open(CSV_FILE, 'r', encoding='utf-8') as f:
                    st.download_button(
                        label="Download Journal CSV",
                        data=f.read(),
                        file_name=f"journal_entries_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No data to export yet.")

if __name__ == "__main__":
    main()