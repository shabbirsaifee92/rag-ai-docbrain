# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from document_processor import DocumentProcessor
from embeddings_manager import EmbeddingsManager
from rag_engine import RAGEngine
from config import settings
import os

app = FastAPI()

class Question(BaseModel):
    text: str

# Initialize components
print("Starting document processing...")
doc_processor = DocumentProcessor()
documents = doc_processor.load_and_split()
print(f"Processed {len(documents)} document chunks")

print("Creating vector store...")
embeddings_manager = EmbeddingsManager()
vectorstore = embeddings_manager.create_vectorstore(documents)
print("Vector store created successfully")

rag_engine = RAGEngine(vectorstore)

@app.post("/ask")
async def ask_question(question: Question):
    try:
        answer = rag_engine.get_answer(question.text)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

