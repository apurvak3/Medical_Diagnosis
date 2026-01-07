import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rbac-diagnosis-index")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Embeddings (local)
embeddings = OllamaEmbeddings(model="llama3:latest")

# LLM (local Ollama)
llm = ChatOllama(
    model="llama3:latest",
    temperature=0
)

prompt = PromptTemplate.from_template("""
You are a medical assistant. Using only the provided context, produce:
1) Probable diagnosis (1–2 lines)
2) Key findings (bullet points)
3) Suggested next steps (clearly mention these are not medical advice)

Context:
{context}

User question:
{question}
""")

rag_chain = prompt | llm


async def diagnosis_report(user: str, doc_id: str, question: str):
    # Embed query
    embedding = await asyncio.to_thread(embeddings.embed_query, question)

    # Query Pinecone
    results = await asyncio.to_thread(
        index.query,
        vector=embedding,
        top_k=5,
        include_metadata=True
    )

    contexts = []
    sources = set()

    for match in results.get("matches", []):
        md = match.get("metadata", {})
        if md.get("doc_id") == doc_id:
            contexts.append(md.get("text", ""))
            sources.add(md.get("source"))

    if not contexts:
        return {"diagnosis": None, "message": "No indexed content found"}

    context_text = "\n\n".join(contexts[:5])

    final = await asyncio.to_thread(
        rag_chain.invoke,
        {"context": context_text, "question": question}
    )

    return {
        "diagnosis": final.content,
        "sources": list(sources)
    }

