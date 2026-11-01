import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Document Intelligence & Semantic Search",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Intelligence & Semantic Search (RAG)")
st.caption("Upload documents, index them into ChromaDB, and query them using Groq LLaMA-3.1 with verified citations.")

# Sidebar: PDF File Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Index into Vector Store", use_container_width=True):
            with st.spinner("Extracting text, chunking, and indexing into ChromaDB..."):
                file_payload = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                }
                try:
                    response = requests.post(f"{API_BASE_URL}/upload", files=file_payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Indexed **{data['filename']}** successfully! ({data['chunks_indexed']} chunks created)")
                    else:
                        st.error(response.json().get("detail", "Upload failed."))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to FastAPI backend. Ensure backend is running on port 8000.")

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

    # Fetch and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching ChromaDB and synthesizing response..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"query": user_question}
                )

                if response.status_code == 200:
                    result = response.json()
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
                else:
                    error_detail = response.json().get("detail", "Failed to retrieve answer.")
                    st.error(f"Error: {error_detail}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI server. Please check that the backend is active on port 8000.")