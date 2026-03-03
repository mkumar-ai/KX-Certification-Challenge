import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from rank_bm25 import BM25Okapi
from retriever import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_SEMANTIC,
    TOP_K_FINAL,
    INDUSTRY_KEYWORDS,
    DOMAIN_KEYWORDS,
    detect_filters,
    build_qdrant_filter,
    bm25_rerank,
)

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
llm           = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


def generate_hypothetical_answer(query: str) -> str:
    """
    Ask LLM to generate a hypothetical answer as if it were from a
    consulting firm capability report. This hypothetical answer uses
    domain-specific terminology matching the internal documents.
    """
    prompt = f"""You are a consultant at a top-tier consulting firm.
A partner has asked: "{query}"

Write a short, realistic paragraph (4-6 sentences) that could appear in an internal 
capability report answering this question. Use specific consulting terminology, 
mention realistic client outcomes, metrics, and technical approaches.
Do NOT say you don't know — generate a plausible, detailed answer."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def embed_text(text: str) -> list:
    """Embed a single text using OpenAI text-embedding-3-small."""
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding


def retrieve_hyde(query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    HyDE retrieval pipeline:
    1. Generate hypothetical answer from query
    2. Embed the hypothetical answer (not the raw query)
    3. Semantic search with metadata pre-filtering
    4. BM25 re-ranking on original query
    """
    # Step 1: Generate hypothetical answer
    print(f"  [HyDE] Generating hypothetical answer...")
    hypothetical_answer = generate_hypothetical_answer(query)
    print(f"  [HyDE] Hypothetical answer: {hypothetical_answer[:120]}...")

    # Step 2: Embed hypothetical answer
    hyde_vector = embed_text(hypothetical_answer)

    # Step 3: Detect metadata filters from original query
    filters       = detect_filters(query)
    qdrant_filter = build_qdrant_filter(filters)

    if filters:
        print(f"  [HyDE] Metadata filters applied: {filters}")
    else:
        print(f"  [HyDE] No metadata filters — full collection search")

    # Step 4: Semantic search using HyDE vector
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=hyde_vector,
        query_filter=qdrant_filter,
        limit=TOP_K_SEMANTIC,
        with_payload=True,
    )
    candidates = results.points
    print(f"  [HyDE] Semantic search returned {len(candidates)} candidates")

    # Step 5: BM25 re-rank using original query
    reranked = bm25_rerank(query, candidates, top_k=top_k)
    print(f"  [HyDE] BM25 re-ranking done. Returning top {len(reranked)} chunks")

    # Step 6: Format output
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
        }
        for hit in reranked
    ]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "I am meeting a CFO worried about regulatory penalties and compliance reporting delays. Do we have a relevant use case?",
        "I need to pitch our fraud detection capabilities. What is the strongest outcome we can talk about?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve_hyde(q)
        for r in results:
            print(f"  -> [{r['use_case_id']}] {r['title']} ({r['capability_domain']})")
