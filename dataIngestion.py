from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import hashlib
import shutil
import os
load_dotenv()

def get_chunk_id(chunk, index):
    content = chunk.page_content + str(index)
    return hashlib.md5(content.encode()).hexdigest()

# Delete old DB first
if os.path.exists("chroma-db"):
    shutil.rmtree("chroma-db")
    print("Old DB deleted!")

data = PyPDFLoader("DocumentLoaders/deeplearning.pdf")
docs = data.load()

# Skip cover/copyright/index pages
docs = docs[20:-10]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# Filter empty chunks
chunks = [c for c in chunks if c.page_content.strip()]

print(f"Total pages: {len(docs)}")
print(f"Total chunks: {len(chunks)}")
print(f"\nSample chunk:\n{chunks[10].page_content}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Unique IDs
ids = [get_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]

vector = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db",
    ids=ids
)

print("\n✅ Vector DB created successfully!")