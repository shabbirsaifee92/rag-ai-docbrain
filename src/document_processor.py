import re
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings

class DocumentProcessor:
    def __init__(self, docs_dir: str = "/app"):
        self.docs_dir = docs_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " "],  # Logical splitting
            keep_separator=False
        )

    def load_and_split(self):
        """
        Load documents, dynamically extract metadata, deduplicate content,
        and split into smaller chunks.
        """
        print(f"Loading document from: {self.docs_dir}/sox.md")
        loader = TextLoader(f"{self.docs_dir}/sox.md")
        documents = loader.load()
        print(f"Found {len(documents)} documents")

        # Deduplicate content and extract metadata
        unique_documents = []
        unique_content = set()

        for doc in documents:
            if doc.page_content not in unique_content:
                unique_content.add(doc.page_content)
                # Extract metadata dynamically from the document structure
                metadata = self._extract_metadata(doc.page_content)
                doc.metadata = metadata
                unique_documents.append(doc)

        print(f"Found {len(unique_documents)} unique documents after deduplication")

        # Split documents into smaller chunks
        chunks = self.text_splitter.split_documents(unique_documents)
        print(f"Split into {len(chunks)} chunks")

        # Remove duplicate chunks to ensure distinct content
        chunks = self.remove_duplicate_chunks(chunks)

        # Debugging: Print a few chunks
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i + 1}: {chunk.page_content[:100]}...")
            print(f"Metadata: {chunk.metadata}")
            print("------------")

        return chunks

    def _extract_metadata(self, text):
        """
        Extract metadata dynamically, such as section titles or headers.
        """
        # Match headers with Markdown-style formatting
        match_h2 = re.search(r'##\s*(.*)', text)  # Secondary header
        match_h3 = re.search(r'###\s*(.*)', text)  # Tertiary header

        section = match_h2.group(1) if match_h2 else "General"
        sub_section = match_h3.group(1) if match_h3 else None

        metadata = {"section": section}
        if sub_section:
            metadata["sub_section"] = sub_section
        return metadata

    def remove_duplicate_chunks(self, chunks):
        """
        Remove duplicate chunks to avoid redundancy in the pipeline.
        """
        seen_contents = set()
        unique_chunks = []

        for chunk in chunks:
            if chunk.page_content not in seen_contents:
                seen_contents.add(chunk.page_content)
                unique_chunks.append(chunk)

        print(f"Reduced to {len(unique_chunks)} unique chunks.")
        return unique_chunks

    def retrieve_chunks(self, query, vector_db):
        """
        Retrieve relevant chunks based on query, without relying on keywords.
        """
        # Convert the query into an embedding
        query_embedding = vector_db.embed_query(query)

        # Retrieve top-k chunks based on semantic similarity
        print("Searching for relevant chunks...")
        retrieved_chunks = vector_db.similarity_search(
            query_embedding=query_embedding,
            k=5  # Number of top results
        )

        # Debug: Log retrieved chunks and metadata
        for chunk in retrieved_chunks:
            print(f"Retrieved Chunk: {chunk.page_content[:100]}...")
            print(f"Metadata: {chunk.metadata}")

        return retrieved_chunks

    def test_complex_query(self, query, vector_db):
        """
        Test a complex query and display the retrieved chunks.
        """
        retrieved_chunks = self.retrieve_chunks(query, vector_db)
        for chunk in retrieved_chunks:
            print(f"Chunk: {chunk.page_content[:100]}...")
            print(f"Metadata: {chunk.metadata}")
            print("------------")
