# document_processor.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings

class DocumentProcessor:
    def __init__(self, docs_dir: str = "/app"):
        self.docs_dir = docs_dir
        # Let RecursiveCharacterTextSplitter handle splitting automatically
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def load_and_split(self):
        print(f"Loading document from: {self.docs_dir}/sox.md")
        loader = TextLoader(f"{self.docs_dir}/sox.md")
        documents = loader.load()
        print(f"Found {len(documents)} documents")

        # Get unique content first
        unique_content = list(set([doc.page_content for doc in documents]))
        print(f"Found {len(unique_content)} unique documents after deduplication")

        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        return chunks
