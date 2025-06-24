from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

loader = CSVLoader(file_path="pu_pages_data.csv", encoding="ISO-8859-1")

# Load and split
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and save vector store
db = Chroma.from_documents(texts, embeddings, persist_directory="./db")
db.persist()

print("Indexing complete with HuggingFaceEmbeddings.")
