import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# 1. Local Hugging Face embeddings running on CPU
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# 2. Local vector database connection
vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model,
    collection_name="rag_documents"
)

# 3. LLM Setup via Groq API
llm = ChatGroq(
    model="openai/gpt-oss-20b",  
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)

def ingest_pdf(file_path: str) -> int:
    """Loads a PDF, creates chunks with overlap, and persists vectors to ChromaDB."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
        )
    chunks = text_splitter.split_documents(documents)

    doc_name = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata["source_doc"] = doc_name

    vector_db.add_documents(chunks)
    return len(chunks)

def format_docs(docs):
    formatted = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        source = doc.metadata.get("source_doc", "Document")
        formatted.append(f"[Page {page_num} | {source}]:\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def answer_query(query: str) -> dict:
    """Retrieves top 4 matching chunks and synthesizes an answer using Groq LLaMA-3.1."""
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return {
            "answer": "No documents indexed yet. Please upload a PDF first.",
            "sources": []
        }

    prompt_template = ChatPromptTemplate.from_template("""
You are an expert Document Intelligence Assistant.
Answer the user's question accurately and strictly using ONLY the context provided below.
If the answer cannot be found in the context, reply with:
"I am sorry, but the provided documents do not contain information to answer this question."
Do not speculate or fabricate information.

Context:
{context}

Question: {question}

Answer:
""")

    rag_chain = (
        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    generated_answer = rag_chain.invoke(query)

    sources = [
        {
            "content": doc.page_content,
            "page": doc.metadata.get("page", 0) + 1,
            "source": doc.metadata.get("source_doc", "Document")
        }
        for doc in retrieved_docs
    ]

    return {
        "answer": generated_answer,
        "sources": sources
    }