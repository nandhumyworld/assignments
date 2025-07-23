# Daily Journaling App 📝

A powerful personal journaling application built with Streamlit and LangChain that helps you maintain daily reflections, plan your day, and analyze your journal entries using AI.

## Features

- **Multiple Journaling Modes:**
  - Reflect briefly
  - Design your day
  - Wrap-Up
  - Ask (AI-powered journal analysis)

- **Smart Vector Search:**
  - Uses FAISS for efficient journal entry search
  - Embeddings powered by OpenAI
  - RAG (Retrieval Augmented Generation) for intelligent responses

- **Data Management:**
  - CSV storage for journal entries
  - Vector database for semantic search
  - Export functionality
  - Session management

## Requirements

```
streamlit
langchain
openai
faiss-cpu
pandas
python-dotenv
```

## Setup

1. Clone the repository
2. Create a `.env` file in the `openai_p` directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Select a date and journaling mode from the sidebar
3. Answer the prompted questions
4. Use the "Ask" mode to analyze your previous entries
5. Export your journal entries as CSV when needed

## Journaling Modes

### Reflect Briefly
- What do I want from today?
- What am I grateful for?
- What will I do if things go wrong?

### Design Your Day
- What must get done today?
- What would make today feel meaningful?
- What could derail me—and how will I respond?

### Wrap-Up
- What did you learn today?
- What did you contribute today?
- What's still on my mind?
- What needs to be carried over to tomorrow?
- One word that captures today is:

### Ask Mode
Use AI to analyze your previous journal entries and gain insights from your past reflections.

## Features in Detail

### Vector Database
- Uses FAISS for efficient similarity search
- Automatically indexes all journal entries
- Enables semantic search across your journal history

### Data Storage
- Entries are stored in CSV format
- Vector embeddings are stored in FAISS index
- Metadata is preserved for context

### AI Integration
- OpenAI embeddings for semantic search
- LangChain for RAG implementation
- Intelligent question answering about your journal entries

## File Structure

- `app.py`: Main application file
- `openai_util.py`: OpenAI API utilities
- `journal_entries.csv`: Journal data storage
- `journal_faiss.index`: FAISS vector index
- `journal_metadata.pkl`: Vector metadata storage

## Contributing

Feel free to submit issues and enhancement requests!
