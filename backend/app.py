import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.rag_engine import ingest_pdf, answer_query

app = FastAPI(
    title="Document Intelligence & Semantic Search API",
    description="RAG Pipeline powered by FastAPI, ChromaDB, and Groq LLaMA-3.1"
)

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryPayload(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "online", "message": "RAG Backend API is operational"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    destination_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(destination_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_indexed = ingest_pdf(destination_path)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": chunks_indexed,
            "message": "File successfully chunked and stored in ChromaDB."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/query")
async def query_documents(payload: QueryPayload):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = answer_query(payload.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")