# Retrieval-Augmented Generation for Technical PDF Question Answering using LangChain and ChromaDB

I built this project to solve a real issue I faced while dealing with heavy technical documentation: normal search tools only match exact words, and standard chatbots tend to hallucinate when asked about internal project details. 

This is a full-stack RAG pipeline that lets you drop in dense technical PDFs, index them into a vector space, and get precise answers backed up with exact source citations.

---

## What It Does & How I Built It

- **Frontend (Streamlit):** Created a clean UI running on port 8501 where anyone can upload a PDF, monitor indexing status, and chat directly with their documents.
- **Backend (FastAPI):** Built a decoupled REST API on port 8000 handling `/upload` and `/query`. Keeping this separate from Streamlit keeps the UI snappy while heavy vector calculations run in the background.
- **Vector Search (ChromaDB):** Handled chunking and embedding storage locally. It performs semantic similarity search so queries pull relevant passages even if keywords don't match word-for-word.
- **LLM Pipeline (LangChain + Groq):** Used LangChain to connect text extraction, ChromaDB retrieval, and prompt injection. The actual answers are generated via Groq using LLaMA-3 with strict instructions to answer only from the provided text or state that the data isn't available.

---

## Project Structure

```text
document-rag-project/
├── backend/
│   ├── app.py              # FastAPI endpoints for document upload and retrieval
│   └── rag_engine.py       # PDF parsing, ChromaDB setup, and query chain logic
├── frontend/
│   └── streamlit_app.py    # Streamlit interface with chat history and source citations
├── .gitignore              # Keeps API keys, virtual environment, and local vector DB safe
├── check_models.py         # Utility script to verify Groq model availability
└── requirements.txt        # Core project dependencies