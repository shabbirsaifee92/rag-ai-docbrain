# rag_engine.py
import requests
from typing import List
from langchain.schema import Document
from llm_client import LLMClient
from config import settings

class RAGEngine:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm_client = LLMClient(base_url=settings.LLM_BASE_URL)

    def _get_relevant_context(self, question: str) -> str:
        docs = self.vectorstore.similarity_search(
            question,
            k=settings.K_DOCUMENTS
        )
        print(f"Found {len(docs)} relevant documents")
        return "\n".join([doc.page_content for doc in docs])

    def get_answer(self, question: str) -> str:
        relevant_context = self._get_relevant_context(question)
        print(f"Retrieved context length: {len(relevant_context)}")

        prompt = f"""<s>[INST] Answer the given question using only the provided context. Be concise and only provide the direct answer to the question asked. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

Context:
{relevant_context}

Question: {question}

Provide a single, focused answer: [/INST]"""

        response = self.llm_client.generate_completion(prompt)
        # Clean up response and take only the first part if it starts answering multiple questions
        answer = response.strip()
        if "\nQuestion:" in answer:
            answer = answer.split("\nQuestion:")[0].strip()

        return answer
