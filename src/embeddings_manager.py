from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import settings

class EmbeddingsManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )

    def create_vectorstore(self, documents):
        print(f"Creating vector store with {len(documents)} documents")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=settings.VECTOR_DB_PATH
        )

    def load_vectorstore(self):
        return Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )
