# rag_pipeline.py
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage, StorageContext
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder

load_dotenv()

app = Flask(__name__)
CORS(app) 

llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_dir = "./chroma_db1"
client = PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("rag-collection")
vector_store = ChromaVectorStore(chroma_collection=collection, persist_dir=persist_dir)

storage_context = StorageContext.from_defaults(
    persist_dir="./index",
    vector_store=vector_store
)
index = load_index_from_storage(storage_context, embed_model=embed_model)

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
    Retrieve relevant docs, build context, prompt LLM and format answer.
    """
    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(query)
    doc_texts = [node.get_content() for node in nodes]

    #top_docs = rerank(query, doc_texts)[:3]
    context = "\n\n---\n\n".join(doc_texts)

    prompt = f"""
You are **PU-Assistant**, the official AI helpdesk chatbot of Panjab University, Chandigarh.

You must strictly answer the student's question **only using the verified information provided below**.
Follow these exact guidelines carefully to keep answers accurate, formal, polite, and helpful:

**Answering rules (read carefully and apply strictly):**
1. If the question is about eligibility, steps, rules, process, fee, or form:
   → answer clearly using neat bullet points (max 4–6).
2. For simple factual or definition-type questions, reply in **one direct, precise sentence**.
3. If the context includes any web page, URL, downloadable form, or PDF:
   → always add it in the answer as a clickable markdown link,
   e.g. [Visit official website](https://example.com)
   → do NOT add any link unless it truly exists in the context.
4. Links must open in a new browser tab when rendered.
5. Never guess, create, or hallucinate a URL or link that is not clearly present in the provided context.
6. If both ₹ (INR) and $ (USD) amounts are found:
   → always mention only the ₹ fee amount.
7. If the answer is genuinely missing:
   → politely respond with one of these:
   > *Sorry, I couldn't find that information. Please contact the university administration.*
   or
   > *Sorry, I couldn't help you with that. Please check the official website.*
   (choose whichever fits better, without mentioning missing data, context, or source).
8. Do NOT mention words like “context”, “data not found”, “source missing” etc.
9. Keep the tone formal, professional, and polite.
10. Avoid repetition; answer directly without unnecessary introduction or disclaimers.

**At the end**:
- Suggest exactly three follow-up questions related to Panjab University admission, fees, process, scholarships, hostels, or campus facilities.
- Each question must be short (max 5–6 words).
- Do NOT repeat the same topic as the user’s original question.
- Do NOT use the same topic twice.
- Format strictly as:
 **Know more about:**
 - Question 1
 - Question 2
 - Question 3

---

**Information you must use**:
{context}

**Student’s Question**:
{query}

 **Your Answer**:
"""

    response = llm.complete(prompt)
    full_text = response.text.strip()

  
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

    pdf_url = None
    if "fee" in query.lower() or "fees" in query.lower():
        pdf_url = "http://127.0.0.1:5000/files/pu_fee_structure.pdf"

    return {"reply": answer_main, "follow_ups": follow_ups, "pdf": pdf_url}


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
