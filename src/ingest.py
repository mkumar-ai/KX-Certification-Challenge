# ingest.py
# Purpose: Parse Capability_Report.docx → build 11 chunks → embed → upsert to Qdrant Cloud

import os
import uuid
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME   = "kx_capabilities"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
DOCX_PATH         = "data/Capability_Report.docx"

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def extract_paragraphs(path: str) -> dict:
    doc = Document(path)
    return {i: p.text.strip() for i, p in enumerate(doc.paragraphs) if p.text.strip()}


def build_chunks(paras: dict) -> list:
    summary_indices = list(range(3, 24))
    summary_text = "\n".join(paras[i] for i in summary_indices if i in paras)

    chunks = [
        {
            "text": summary_text,
            "use_case_id": "summary",
            "title": "Executive Summary & Portfolio Overview",
            "capability_domain": "All",
            "industry": ["Banking", "Insurance", "Healthcare", "Financial Services"],
            "client": "Multiple",
            "type": "summary",
            "source": "internal",
        }
    ]

    use_case_meta = [
        {"start": 31, "end": 42, "use_case_id": "UC1", "title": "AI-Powered Credit Risk Scoring", "capability_domain": "AI & Machine Learning", "industry": ["Banking"], "client": "BankNova"},
        {"start": 42, "end": 52, "use_case_id": "UC2", "title": "Real-Time Fraud Detection", "capability_domain": "AI & Machine Learning", "industry": ["Banking"], "client": "NexBank"},
        {"start": 52, "end": 63, "use_case_id": "UC3", "title": "Customer Churn Prediction & Retention Engine", "capability_domain": "AI & Machine Learning", "industry": ["Banking"], "client": "CrescentBank"},
        {"start": 63, "end": 74, "use_case_id": "UC4", "title": "Intelligent Claims Automation", "capability_domain": "AI & Machine Learning", "industry": ["Insurance"], "client": "ShieldGen Insurance"},
        {"start": 79, "end": 90, "use_case_id": "UC5", "title": "Enterprise Data Lakehouse Migration", "capability_domain": "Data Engineering", "industry": ["Banking"], "client": "HorizonBank"},
        {"start": 90, "end": 100, "use_case_id": "UC6", "title": "Basel III Regulatory Reporting Automation", "capability_domain": "Data Engineering", "industry": ["Banking"], "client": "EuroCapital Bank"},
        {"start": 100, "end": 110, "use_case_id": "UC7", "title": "Unified Customer Data Platform (Customer 360)", "capability_domain": "Data Engineering", "industry": ["Banking"], "client": "PivotBank"},
        {"start": 115, "end": 130, "use_case_id": "UC8", "title": "Zero-Trust Architecture Implementation", "capability_domain": "Cyber Security", "industry": ["Financial Services"], "client": "ApexCapital Advisors"},
        {"start": 130, "end": 141, "use_case_id": "UC9", "title": "AI-Powered SOC Transformation", "capability_domain": "Cyber Security", "industry": ["Healthcare"], "client": "MedShield Health System"},
        {"start": 146, "end": 160, "use_case_id": "UC10", "title": "End-to-End Retail Banking Digital Transformation", "capability_domain": "Digital Transformation", "industry": ["Banking"], "client": "FutureBank"},
    ]

    for uc in use_case_meta:
        text = "\n".join(paras[i] for i in range(uc["start"], uc["end"]) if i in paras)
        chunks.append({
            "text": text,
            "use_case_id": uc["use_case_id"],
            "title": uc["title"],
            "capability_domain": uc["capability_domain"],
            "industry": uc["industry"],
            "client": uc["client"],
            "type": "use_case",
            "source": "internal",
        })

    return chunks


def embed_texts(texts: list) -> list:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ensure_collection():
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        for field, schema in [("capability_domain", PayloadSchemaType.KEYWORD), ("industry", PayloadSchemaType.KEYWORD), ("type", PayloadSchemaType.KEYWORD)]:
            qdrant_client.create_payload_index(collection_name=COLLECTION_NAME, field_name=field, field_schema=schema)
        print(f"Collection '{COLLECTION_NAME}' created with payload indexes.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")


def upsert_chunks(chunks: list, embeddings: list):
    points = []
    for chunk, vector in zip(chunks, embeddings):
        payload = {**chunk}
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} chunks to '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    print("Parsing document...")
    paras = extract_paragraphs(DOCX_PATH)

    print("Building chunks...")
    chunks = build_chunks(paras)
    print(f"  -> {len(chunks)} chunks built")
    for c in chunks:
        print(f"  [{c['use_case_id']}] {c['title']} | {len(c['text'])} chars")

    print("\nEmbedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  -> {len(embeddings)} embeddings generated")

    print("\nEnsuring Qdrant collection...")
    ensure_collection()

    print("\nUpserting to Qdrant...")
    upsert_chunks(chunks, embeddings)

    print("\nIngestion complete!")