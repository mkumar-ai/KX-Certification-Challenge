# retriever.py
# Purpose: Hybrid retrieval — Metadata Pre-filter + Semantic Vector Search + BM25 Re-ranking

import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from rank_bm25 import BM25Okapi

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "kx_capabilities"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_SEMANTIC  = 8
TOP_K_FINAL     = 4
MAX_COLLECTION  = 11   # total chunks in collection — hard cap for number requests

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── Metadata keyword maps ─────────────────────────────────────────────────────
INDUSTRY_KEYWORDS = {
    "Banking":            ["banking", "bank", "retail bank", "commercial bank"],
    "Insurance":          ["insurance", "insurer", "claims"],
    "Healthcare":         ["healthcare", "health system", "hospital", "medical"],
    "Financial Services": ["financial services", "investment", "asset management", "wealth"],
}

DOMAIN_KEYWORDS = {
    "AI & Machine Learning":  ["ai", "machine learning", "ml", "fraud", "churn", "credit", "scoring", "claims automation", "predictive"],
    "Data Engineering":       ["data engineering", "lakehouse", "data platform", "etl", "regulatory reporting", "customer 360", "basel", "cdp"],
    "Cyber Security":         ["cyber", "security", "zero trust", "soc", "threat", "ransomware", "siem"],
    "Digital Transformation": ["digital transformation", "core banking", "modernisation", "digitisation"],
}


def detect_filters(query: str) -> dict:
    q = query.lower()
    detected = {}

    matched_industries = [
        industry for industry, keywords in INDUSTRY_KEYWORDS.items()
        if any(kw in q for kw in keywords)
    ]
    if matched_industries:
        detected["industry"] = matched_industries

    matched_domains = [
        domain for domain, keywords in DOMAIN_KEYWORDS.items()
        if any(kw in q for kw in keywords)
    ]
    if matched_domains:
        detected["capability_domain"] = matched_domains[0]

    return detected


def build_qdrant_filter(filters: dict):
    if not filters:
        return None
    conditions = []
    if "industry" in filters:
        conditions.append(FieldCondition(key="industry", match=MatchAny(any=filters["industry"])))
    if "capability_domain" in filters:
        conditions.append(FieldCondition(key="capability_domain", match=MatchValue(value=filters["capability_domain"])))
    return Filter(must=conditions) if conditions else None


def embed_query(query: str) -> list:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def semantic_search(query_vector: list, qdrant_filter, top_k: int) -> list:
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )
    return results.points


def bm25_rerank(query: str, candidates: list, top_k: int) -> list:
    if not candidates:
        return []
    texts = [hit.payload["text"] for hit in candidates]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    k = 60
    semantic_ranks = {i: i + 1 for i in range(len(candidates))}
    bm25_ranks = {
        i: rank + 1
        for rank, i in enumerate(
            sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)
        )
    }
    rrf_scores = {
        i: (1 / (k + semantic_ranks[i])) + (1 / (k + bm25_ranks[i]))
        for i in range(len(candidates))
    }
    ranked = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [candidates[i] for i in ranked[:top_k]]


def retrieve(query: str, top_k: int = TOP_K_FINAL) -> list:
    # Cap top_k at max collection size to prevent excessive retrieval
    top_k = min(top_k, MAX_COLLECTION)

    filters = detect_filters(query)
    qdrant_filter = build_qdrant_filter(filters)

    if filters:
        print(f"  [Retriever] Metadata filters applied: {filters}")
    else:
        print(f"  [Retriever] No metadata filters — full collection search")

    query_vector = embed_query(query)
    candidates   = semantic_search(query_vector, qdrant_filter, top_k=TOP_K_SEMANTIC)
    print(f"  [Retriever] Semantic search returned {len(candidates)} candidates")

    reranked = bm25_rerank(query, candidates, top_k=top_k)
    print(f"  [Retriever] BM25 re-ranking done. Returning top {len(reranked)} chunks")

    return [
        {
            "text":               hit.payload["text"],
            "use_case_id":        hit.payload.get("use_case_id"),
            "title":              hit.payload.get("title"),
            "capability_domain":  hit.payload.get("capability_domain"),
            "industry":           hit.payload.get("industry"),
            "client":             hit.payload.get("client"),
            "type":               hit.payload.get("type"),
            "source":             hit.payload.get("source"),
            "delivery_year":      hit.payload.get("delivery_year", "2024-2026"),
        }
        for hit in reranked
    ]


if __name__ == "__main__":
    test_queries = [
        "What AI and ML use cases have we delivered in Banking?",
        "How many use cases have we delivered in total?",
        "Tell me about fraud detection work done for banks",
        "What cybersecurity engagements have we completed?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q)
        for r in results:
            print(f"  -> [{r['use_case_id']}] {r['title']} ({r['capability_domain']} | {r['delivery_year']})")
