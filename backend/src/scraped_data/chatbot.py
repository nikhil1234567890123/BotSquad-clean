from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# 🔹 Load environment variables (make sure you have a .env file with HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# 🔹 Load embedding model and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./db", embedding_function=embedding_model)

# 🔹 Set up retriever from the vector DB
retriever = db.as_retriever()

# 🔹 Load HuggingFace LLM (you can change model if needed)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.3, "max_length": 512}
)

# 🔹 Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🔹 User query loop
print("🔗 PU Admission Chatbot is ready! Ask anything or type 'exit'.")
while True:
    query = input("\n❓ Ask a PU Admission Question (or type 'exit'): ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break
    try:
        result = qa_chain.run(query)
        print("🤖", result)
    except Exception as e:
        print("❌ Error:", e)
