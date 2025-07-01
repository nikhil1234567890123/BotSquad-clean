# generate_answer.py

import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# 🔹 LLM: Mixtral via Groq
llm = Groq(api_key=groq_api_key, model="llama3-8b-8192")

# 🔹 Embedding: Use local HuggingFace model (no OpenAI needed)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # lightweight + accurate

def generate_answer(query, context):
    doc = Document(text=context)

    index = VectorStoreIndex.from_documents([doc], embed_model=embed_model)

    # 🔹 Custom Prompt
    qa_prompt_template = PromptTemplate(
        """You are a helpful assistant for students exploring Panjab University.

Answer the question in clear bullet points based ONLY on the context below.

Context:
{context_str}

Question: {query_str}
Answer:"""
    )

    query_engine = index.as_query_engine(llm=llm, text_qa_template=qa_prompt_template)
    response = query_engine.query(query)

    return str(response)