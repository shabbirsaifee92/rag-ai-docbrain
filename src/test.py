# test_vector.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load document
loader = TextLoader("product_docs.md")
documents = loader.load()
print(f"Documents loaded: {len(documents)}")

# 2. Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)
print(f"Chunks created: {len(chunks)}")

# 3. Create vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 4. Test search with similarity scores
query = "What are the system requirements?"
results = vectorstore.similarity_search_with_score(query, k=4)

print("\nFiltered Results (score > 0):")
relevant_results = [
    (doc, score) for doc, score in results
    if score > 0  # Only keep positive similarity scores
]

for i, (doc, score) in enumerate(relevant_results):
    print(f"\nResult {i+1} (similarity: {score:.4f}):")
    print(doc.page_content)
