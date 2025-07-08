#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage, StorageContext
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder

# Initialize LLM
llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistent storage setup
persist_dir = "./chroma_db2"
client = PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("rag-collection")
vector_store = ChromaVectorStore(chroma_collection=collection, persist_dir=persist_dir)

# Load vector index
storage_context = StorageContext.from_defaults(
    persist_dir="./index",
    vector_store=vector_store
)
index = load_index_from_storage(storage_context, embed_model=embed_model)

# Load re-ranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, docs):
    """
    Re-rank docs based on relevance to the query using CrossEncoder.
    """
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return sorted_docs

def generate_answer(query):
    """
    Generate answer from the index and LLM.
    """
    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(query)
    doc_texts = [node.get_content() for node in nodes]

    #top_docs = rerank(query, doc_texts)[:3]

    context = "\n\n---\n\n".join(doc_texts)  # Use top 3 docs directly for simplicity)

    prompt = f"""
You are **PU-Assistant**, the official virtual helpdesk for Panjab University, Chandigarh.

Please answer the student's question **strictly using only the information provided below**.

**Answering Rules:**
- If the question is about eligibility, process, fee, or form → reply in **brief bullet points**.
- If there’s a useful **URL or downloadable form**, mention it politely at the end.
- For simple factual or definition questions → reply in **one clear sentence**.
- If the answer isn’t in the provided info → politely say:
   > "Sorry, I couldn't find that information. You may contact the university administration."
- Never mention "context", "source", "data not found", or talk about missing info.
- Always use only ₹ amounts if both ₹ and $ are present.
- Use **formal, polite tone**.
- End with:
   **Know more about:**
   - (follow-up question 1)
   - (follow-up question 2)
   - (follow-up question 3)

ℹ **Information**:
{context}

Question: {query}

Answer:
"""

    response = llm.complete(prompt)
    print("\n--- Answer ---\n")
    print(response.text.strip())
    print("\n--------------\n")

if __name__ == "__main__":
    print("✅ PU-Assistant Chatbot ready. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting. Goodbye!")
            break
        if query:
            try:
                generate_answer(query)
            except Exception as e:
                print(f"❌ Error: {e}")
