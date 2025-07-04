# rag_pipeline.py
import os
from dotenv import load_dotenv
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage, StorageContext
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()

# Initialize LLM
llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistent vector DB setup
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

# Initialize re-ranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, docs):
    """
    Re-rank retrieved documents based on query relevance.
    """
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return sorted_docs

def generate_answer(query):
    """
    Retrieve relevant docs, generate answer & extract follow-up suggestions.
    """
    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(query)
    doc_texts = [node.get_content() for node in nodes]

    top_docs = rerank(query, doc_texts)[:3]
    context = "\n\n---\n\n".join(top_docs)

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
    full_text = response.text.strip()

    # Extract follow-up lines
    follow_ups = []
    if "**Know more about:**" in full_text:
        parts = full_text.split("**Know more about:**")
        answer_main = parts[0].strip()
        follow_lines = parts[1].strip().splitlines()
        follow_ups = [
            line.replace("-", "").replace("•", "").strip()
            for line in follow_lines if line.strip()
        ]
    else:
        answer_main = full_text

    return {"reply": answer_main, "follow_ups": follow_ups}
