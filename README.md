# Retrieval-Augmented Generation for Technical PDF Question Answering using LangChain and ChromaDB

**Live Demo:** [https://rag-technical-pdf.streamlit.app/](https://rag-technical-pdf.streamlit.app/)

I built this project to solve a real issue I faced while dealing with heavy technical documentation: normal search tools only match exact words, and standard chatbots tend to hallucinate when asked about internal project details.

This is a full-stack RAG pipeline that lets you drop in dense technical PDFs, index them into a vector space, and get precise answers backed up with exact page-level source citations.

---

## What It Does & How I Built It

- **Frontend (Streamlit):** Created a clean UI running on port 8501 where anyone can upload a PDF, monitor indexing status, and chat directly with their documents.
- **Backend (FastAPI):** Built a decoupled REST API on port 8000 handling `/upload` and `/query`. Keeping this separate from Streamlit keeps the UI snappy while heavy vector calculations run in the background.
- **Cloud Entrypoint (`cloud_app.py`):** Added a standalone wrapper that connects the Streamlit frontend directly to the RAG engine so the app runs smoothly on Streamlit Community Cloud without needing a separate background server.
- **Vector Search (ChromaDB & HuggingFace):** Used `sentence-transformers/all-MiniLM-L6-v2` to generate local CPU embeddings and stored them in ChromaDB. It runs semantic similarity search (`k=6`) so queries pull relevant passages even when keywords don't match word-for-word.
- **LLM Pipeline (LangChain + Groq):** Used LangChain to connect PDF text extraction, recursive chunking (`1200` chunk size with `200` overlap), ChromaDB retrieval, and prompt injection. Answers are generated via Groq using strict instructions to answer only from the provided text or state that the data isn't available.

---

## Project Structure

```text
rag-technical-pdf-qa/
├── backend/
│   ├── app.py              # FastAPI endpoints for document upload and retrieval
│   └── rag_engine.py       # PDF parsing, ChromaDB setup, and query chain logic
├── frontend/
│   └── streamlit_app.py    # Streamlit interface for local FastAPI connection
├── cloud_app.py            # Standalone entrypoint for Streamlit Community Cloud
├── .gitignore              # Keeps API keys, virtual environment, and local vector DB safe
├── check_models.py         # Utility script to verify Groq model availability
├── README.md               # Project documentation
└── requirements.txt        # Core project dependencies