# pdf_to_index.py

import os
import json
import pandas as pd
import re
import neattext.functions as nfx
from tqdm import tqdm
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from chromadb import PersistentClient

# 1. Load and flatten JSON data
def load_json_to_dataframe(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for doc in raw_data:
        for page in doc["content"]:
            records.append({
                "pdf_file": doc["pdf_file"],
                "page_number": page["page_number"],
                "text": page.get("text", "").strip(),
                "tables": page.get("tables", [])
            })

    return pd.DataFrame(records)

# 2. Clean text content
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = nfx.remove_multiple_spaces(text)
    text = re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z0-9\-\(\)\s]', ' ', text))
    return text.strip()

# 3. Flatten tables into readable format
def flatten_tables_verbose(tables) -> str:
    if not tables or not isinstance(tables, list):
        return ""
    
    flat = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        headers = table[0]
        for row in table[1:]:
            row_text = ", ".join(
                f"{headers[i]}: {cell.strip() if cell else ''}"
                for i, cell in enumerate(row) if i < len(headers)
            )
            flat.append(row_text)
    
    return " ".join(flat)

# 4. Chunk text based on token length
def chunk_text_token_based(text, tokenizer, max_tokens=512, stride=50):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk)
        start += max_tokens - stride
    return chunks

# 5. Main processing and indexing function
def process_and_index(filepath: str, tokenizer):
    # Load and clean data
    df = load_json_to_dataframe(filepath)
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['tables_text'] = df['tables'].apply(flatten_tables_verbose)
    
    # Chunk the text and table data
    df['chunks'] = df['cleaned_text'].apply(lambda x: chunk_text_token_based(x, tokenizer))
    df['chunks_table'] = df['tables_text'].apply(lambda x: chunk_text_token_based(x, tokenizer))

    # Create embedding model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Setup Chroma vector store
    persist_dir = "./chroma_db2"
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("rag-collection")
    vector_store = ChromaVectorStore(chroma_collection=collection, persist_dir=persist_dir)

    # Storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create Documents
    documents = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="📄 Creating documents"):
        for i, chunk in enumerate(row['chunks']):
            documents.append(Document(
                text=chunk,
                metadata={
                    "source": row["pdf_file"],
                    "chunk_type": "text",
                    "chunk_id": f"text_{i}"
                }
            ))
        for j, chunk in enumerate(row['chunks_table']):
            documents.append(Document(
                text=chunk,
                metadata={
                    "source": row["pdf_file"],
                    "chunk_type": "table",
                    "chunk_id": f"table_{j}"
                }
            ))

    # Parse and embed documents
    parser = SimpleNodeParser()
    nodes = []
    for doc in tqdm(documents, desc="📄 Parsing documents"):
        nodes.extend(parser.get_nodes_from_documents([doc]))

    for node in tqdm(nodes, desc="🔍 Embedding nodes"):
        node.embedding = embed_model.get_text_embedding(node.text)

    # Build index
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # Persist the index
    index.storage_context.persist(persist_dir="./index2")

    print(f"\n✅ Index created and persisted with {len(nodes)} nodes.")

# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    process_and_index("pdf_data.json", tokenizer)
