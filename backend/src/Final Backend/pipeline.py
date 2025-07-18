import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import csv
from datetime import datetime

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder

# ---------- FLASK APP INITIALIZATION ----------

load_dotenv()
app = Flask(__name__)
CORS(app)

# ---------- COMPONENT INITIALIZATION ----------

def init_llm():
    return Groq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5
    )

def init_embed_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def init_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def init_vector_store(persist_dir="./chroma_db1", collection_name="rag-collection"):
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    return ChromaVectorStore(chroma_collection=collection, persist_path=persist_dir)

def load_index(persist_dir="./chroma_db1", index_dir="./index", embed_model=None, vector_store=None):
    storage_context = StorageContext.from_defaults(
        persist_dir=index_dir,
        vector_store=vector_store
    )
    return load_index_from_storage(storage_context, embed_model=embed_model)

llm = init_llm()
embed_model = init_embed_model()
reranker = init_reranker()
vector_store = init_vector_store()
index = load_index(embed_model=embed_model, vector_store=vector_store)

# ---------- SESSION MEMORY ----------
session = {
    "original_query": None,
    "expecting_clarification": False
}

# ---------- RERANKING ----------
def rerank(query, docs, reranker_model):
    pairs = [[query, doc] for doc in docs]
    scores = reranker_model.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return sorted_docs

# ---------- MAIN FUNCTION ----------
def generate_answer(query):
    global session

    if session["expecting_clarification"] and session["original_query"]:
        query = f"{session['original_query']} for {query}"
        session["original_query"] = None
        session["expecting_clarification"] = False
    else:
        vague_keywords = ["fee", "admission", "form", "hostel", "apply", "scholarship", "process"]
        if any(keyword in query.lower() for keyword in vague_keywords):
            session["original_query"] = query
            session["expecting_clarification"] = True

    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(query)
    doc_texts = [node.get_content() for node in nodes]
    context = "\n\n---\n\n".join(doc_texts)

    prompt = f"""
You are *PU-Assistant*, the official AI helpdesk chatbot of Panjab University, Chandigarh.

You must answer the student's query *strictly using the verified information provided below*. Do not use any external knowledge. Follow all the guidelines carefully to ensure your responses are accurate, formal, polite, and helpful.

*Answering Rules (apply strictly):*
1. If the question is about eligibility, admission steps, rules, process, fees, or forms:
   → Answer clearly using neat bullet points (maximum 4–6 points).
2. For simple factual or definition-style questions:
   → Reply in *one direct, precise sentence*.
3. If any web page, downloadable form, or PDF is mentioned in the context:
   → Include it in the response as a clickable markdown link,  
     e.g. [Visit official website](https://example.com)
   → Only include links that are clearly present in the context.
4. All links must open in a new browser tab.
5. Never guess, assume, or generate a URL or link not found in the context.
6. If both ₹ (INR) and $ (USD) are mentioned:
   → Mention only the ₹ (INR) amount in the answer.
7. If the required information is not found:
   → Respond politely with:
   > Sorry, I couldn't find that information. Please contact the university administration.  
   or  
   > Sorry, I couldn't help you with that. Please check the official website.  
8. Do NOT use phrases like “context not available” or “data not found”.
9. Maintain a formal, professional, and polite tone throughout.
10. Avoid repetition and unnecessary introductions.

*Smart Clarification Logic (very important):*
- If the student's query is vague, generic, or incomplete (e.g., “fee structure”, “admission”, “apply”, “form”, “hostel”, “scholarship”, etc.), then:
   → DO NOT attempt to answer directly.
   → Politely ask for clarification by providing 2–3 relevant options.
   → Format it like:
     > Could you please clarify which of the following you're referring to?  
     > - Option 1  
     > - Option 2  
     > - Option 3

*Follow-Up Suggestions (only after giving a complete answer):*
*Know more about:*
 - Question 1  
 - Question 2  
 - Question 3

---

*Verified Information You Must Use:*
{context}

*Student’s Question:*
{query}

*Your Answer:*
"""

    response = llm.complete(prompt)
    full_text = response.text.strip()

    follow_ups = []
    if "*Know more about:*" in full_text:
        parts = full_text.split("*Know more about:*")
        answer_main = parts[0].strip()
        follow_lines = parts[1].strip().splitlines()
        follow_ups = [
            line.replace("-", "").replace("•", "").strip()
            for line in follow_lines if line.strip()
        ]
    else:
        answer_main = full_text

    pdf_url = None
    if "fee" in query.lower() or "fees" in query.lower():
        pdf_url = "http://127.0.0.1:5000/files/pu_fee_structure.pdf"

    return {"reply": answer_main, "follow_ups": follow_ups, "pdf": pdf_url}

# ---------- FLASK ROUTES ----------

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('message', '')
    answer = generate_answer(user_query)
    return jsonify(answer)

@app.route('/files/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
