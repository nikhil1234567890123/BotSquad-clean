import os
from dotenv import load_dotenv
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient

load_dotenv()

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag-collection"
DATA_DIR = "./data"

print("📦 Loading documents from:", DATA_DIR)
documents = SimpleDirectoryReader(DATA_DIR).load_data()

print("🧠 Initializing embedding model...")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("📚 Creating Chroma collection...")
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection, persist_path=PERSIST_DIR)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("📌 Creating index...")
index = GPTVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

print("💾 Persisting index to:", PERSIST_DIR)
index.storage_context.persist()
print("✅ Indexing complete.")
