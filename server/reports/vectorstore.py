#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import os, time, asyncio
from fastapi import UploadFile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..config.db import reports_collection
from pinecone import Pinecone, ServerlessSpec

embed_model = OllamaEmbeddings(model="mistral")
# Load env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rbac-diagnosis-index")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric="dotproduct", spec=spec)
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# init Ollama embeddings
embed_model = OllamaEmbeddings(model="llama3:latest")# change model as per your pull


async def load_vectorstore(uploaded_files: List[UploadFile], uploaded: str, doc_id: str):
    """
    Save files, chunk texts, embed texts, upsert in Pinecone and write metadata to Mongo
    """

    for file in uploaded_files:
        filename = Path(file.filename).name
        save_path = Path(UPLOAD_DIR) / f"{doc_id}_{filename}"
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

    # load pdf pages
    loader = PyPDFLoader(str(save_path))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    texts = [chunk.page_content for chunk in chunks]
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": filename,
            "doc_id": doc_id,
            "uploader": uploaded,
            "page": chunk.metadata.get("page", None),
            "text": chunk.page_content[:2000],
        }
        for chunk in chunks
    ]

    # get embeddings (Ollama runs locally, so no need async-to-thread)
    embeddings = embed_model.embed_documents(texts)

    # upsert to Pinecone
    def upsert():
        index.upsert(vectors=list(zip(ids, embeddings, metadatas)))

    await asyncio.to_thread(upsert)

    # save report metadata in Mongo
    reports_collection.insert_one({
        "doc_id": doc_id,
        "filename": filename,
        "uploader": uploaded,
        "num_chunks": len(chunks),
        "uploaded_at": time.time()
    })
