from dotenv import load_dotenv
load_dotenv()

import traceback

from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
db = Chroma(
    persist_directory="./db",
    embedding_function=embedding_model
)

# Print total documents in the vector store
print(f"\n Total Documents in Vector DB: {len(db.get()['ids'])}")

# Create retriever
retriever = db.as_retriever()

# Initialize a supported HuggingFace LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base", 
    temperature=0.3,
    max_new_tokens=512
)

# Create a Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Chat loop
while True:
    query = input("\n Ask a PU Admission Question (or type 'exit'): ").strip()
    if query.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    try:
        result = qa_chain.invoke({"query": query})
        print("🤖", result["result"])
    except Exception as e:
        print("Error occurred while processing your question:")
        traceback.print_exc()
