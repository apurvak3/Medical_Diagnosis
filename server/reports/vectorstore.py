from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import os, time
from fastapi import UploadFile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from ..config.db import reports_collection

# ================== CONFIG ==================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rbac-diagnosis-index")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================== EMBEDDINGS ==================
embed_model = OllamaEmbeddings(model="llama3")

# ================== PINECONE ==================
pc = Pinecone(api_key=PINECONE_API_KEY)


def get_pinecone_index():
    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )

        # wait until ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)


index = get_pinecone_index()

# ================== MAIN FUNCTION ==================
async def load_vectorstore(
    uploaded_files: List[UploadFile],
    uploaded: str,
    doc_id: str
):
    for file in uploaded_files:
        filename = Path(file.filename).name
        save_path = Path(UPLOAD_DIR) / f"{doc_id}_{filename}"

        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        loader = PyPDFLoader(str(save_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        texts = [c.page_content for c in chunks]
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]

        metadatas = [{
            "source": filename,
            "doc_id": doc_id,
            "uploader": uploaded,
            "page": c.metadata.get("page")
        } for c in chunks]

        embeddings = embed_model.embed_documents(texts)

        vectors = list(zip(ids, embeddings, metadatas))
        index.upsert(vectors=vectors)

        reports_collection.insert_one({
            "doc_id": doc_id,
            "filename": filename,
            "uploader": uploaded,
            "chunks": len(chunks),
            "uploaded_at": time.time()
        })

    return {"status": "success"}
