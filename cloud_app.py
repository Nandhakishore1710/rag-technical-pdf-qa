import os
import sys
import tempfile

# Fix SQLite version issue for ChromaDB on Streamlit Cloud (Linux)
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

# Bridge Streamlit Cloud Secrets into environment variables before importing rag_engine
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from backend.rag_engine import ingest_pdf, answer_query, vector_db

st.set_page_config(
    page_title="Document Intelligence & Semantic Search",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Intelligence & Semantic Search (RAG)")
st.caption("Upload documents, index them into ChromaDB, and query them using Groq LLaMA-3 with verified citations.")

# Sidebar: PDF File Upload & Database Reset
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Index into Vector Store", use_container_width=True):
            with st.spinner("Extracting text, chunking, and indexing into ChromaDB..."):
                try:
                    # Save uploaded file temporarily with its original filename
                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Directly call rag_engine
                    chunks_count = ingest_pdf(temp_file_path)
                    st.success(f"Indexed **{uploaded_file.name}** successfully! ({chunks_count} chunks created)")

                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except Exception as e:
                    st.error(f"Error indexing document: {str(e)}")

    st.divider()
    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        try:
            existing_data = vector_db.get()
            if existing_data and existing_data.get("ids"):
                vector_db.delete(ids=existing_data["ids"])
            st.session_state.chat_history = []
            st.success("Vector database and chat history cleared!")
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")

# Session state: initialize conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render existing chat conversation
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("🔍 View Retrieved Document Sources"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}:** {src['source']} (Page {src['page']})")
                    st.code(src["content"], language="text")

# Chat input prompt
if user_question := st.chat_input("Ask a question about your uploaded documents..."):
    # Display user query
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Fetch and display assistant response directly via rag_engine
    with st.chat_message("assistant"):
        with st.spinner("Searching ChromaDB and synthesizing response..."):
            try:
                result = answer_query(user_question)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("🔍 View Retrieved Document Sources"):
                        for idx, src in enumerate(sources):
                            st.markdown(f"**Source {idx+1}:** {src['source']} (Page {src['page']})")
                            st.code(src["content"], language="text")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")