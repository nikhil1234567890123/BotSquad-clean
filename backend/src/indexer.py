from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
import os
documents = SimpleDirectoryReader(input_files=["pu_admissions.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./index")
print("Index created and saved!")
