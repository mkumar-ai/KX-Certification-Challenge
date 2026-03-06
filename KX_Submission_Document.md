# KX — AI-Powered Knowledge Assistant

**Submitted by:** Manish Kumar  
**Cohort:** AIM #09, AI Engineering Bootcamp 2026

> **Note:** Nexvance Consulting Group is a fictional company created for this certification challenge.

---

## 1 — Problem & Audience

### Problem Statement

Consulting firm partners cannot quickly surface relevant past use cases before client meetings — creating team dependency, preparation delays, and missed business development opportunities.

### Why This Is a Problem

Partners operate in a high-stakes, time-pressured environment. Before meeting a CFO worried about regulatory compliance or a CISO evaluating zero-trust architecture, they need to instantly recall what the firm has delivered for similar clients. Currently this knowledge is locked in long-form documents, scattered across shared drives, and accessible only through colleagues. The partner either spends significant time searching manually or walks into the meeting under-prepared — directly impacting win rates and client confidence.

When internal experience is limited in a sector, partners also have no way to supplement with latest market use cases — a gap KX addresses by combining internal retrieval with real-time web search.

### Target User

Partners, Directors, and Senior Managers at Nexvance Consulting Group in client-facing roles responsible for business development and pre-meeting preparation.

### Evaluation Questions (Input-Output Pairs)

| # | Question | Expected Output |
|---|----------|-----------------|
| 1 | What AI and ML use cases have we delivered in Banking? | Internal — UC1, UC2, UC3 |
| 2 | How many use cases have we delivered in total? | Internal — summary count |
| 3 | Do we have experience in the Insurance sector? | Internal — UC4 |
| 4 | What cybersecurity engagements have we completed? | Internal — UC8, UC9 |
| 5 | Latest fraud detection use cases in the market in 2025? | Internal + Web Search |
| 6 | Meeting a CFO worried about regulatory compliance — relevant use case? | Internal — UC6 |
| 7 | Pitch our fraud detection capabilities to a prospect. | Internal — UC2 with metrics |
| 8 | Do we have Healthcare experience? | Internal — UC9 |
| 9 | Meeting a retail bank CDO — which use case to lead with? | Internal — UC10 |
| 10 | How many total use cases and across which capability areas? | Internal — summary |
| 11 | Prospect CISO asking about security posture for a financial firm. | Internal — UC8 |
| 12 | Experience helping banks reduce churn? | Internal — UC3 |
| 13 | What Data Engineering use cases have we delivered? | Internal — UC5, UC6, UC7 |

---

## 2 — Proposed Solution

### Solution Overview

KX is an Agentic RAG application that combines private internal knowledge retrieval with real-time web search. A LangGraph-orchestrated agent searches internal capability documents first, assesses sufficiency, and only triggers Tavily web search when external market intelligence is needed or internal results fall short.

### Infrastructure Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| LLM (Generation) | GPT-4o | Best quality for structured partner-facing responses |
| LLM (Classifier) | GPT-4o-mini | Fast, cheap intent classification and scope check |
| Agent Orchestration | LangGraph | Explicit node-based control flow with conditional routing |
| Web Search | Tavily Search API | Purpose-built for AI agents; returns structured results with URLs |
| Embedding Model | text-embedding-3-small | 1536-dim; best cost-to-quality for semantic search |
| Vector Database | Qdrant Cloud | Payload indexing for metadata pre-filtering |
| Monitoring | LangSmith | End-to-end observability for every agent run |
| Evaluation | RAGAS | Automated faithfulness, precision, recall, and relevancy scoring |
| UI | Streamlit | ChatGPT-style chat interface |
| Deployment | Render | Git-based auto-deploy |

### RAG & Agent Components

**RAG Pipeline:**
- `Capability_Report.docx` parsed into 11 semantic chunks (1 summary + 10 use cases)
- Embedded with `text-embedding-3-small` (1536-dim vectors)
- Stored in Qdrant Cloud with payload indexes on `industry`, `capability_domain`, `type`
- Hybrid retrieval: metadata pre-filter → semantic search → BM25 re-ranking (RRF)
- Advanced retriever: HyDE — hypothetical answer generated before embedding

**Agent Nodes:**
- **Node 0 — Scope Check:** LLM classifier rejects non-consulting queries
- **Node 1 — Internal Retrieval:** Hybrid retriever against Qdrant
- **Node 2 — Sufficiency Assessment:** Deterministic rules — external keywords or result shortfall triggers web search
- **Node 3 — Web Search (conditional):** Tavily API, triggered only when needed
- **Node 4 — Answer Generation:** GPT-4o synthesises internal + web context

---

## 3 — Data & API Access

### Chunking Strategy

**Section-Based Chunking** — one chunk per use case (10 use cases + 1 Executive Summary = 11 chunks total). Each chunk contains the full Business Context, Solution, and Results as one contiguous block — preserving context and enabling self-contained retrieval. The Executive Summary chunk handles aggregate queries like *"how many use cases in total"*.

### Metadata per Chunk

| Field | Example | Purpose |
|-------|---------|---------|
| use_case_id | UC1, summary | Unique identifier |
| title | AI-Powered Credit Risk Scoring | Human-readable label |
| capability_domain | AI & Machine Learning | Domain pre-filter |
| industry | [Banking, Insurance] | Industry pre-filter |
| client | BankNova | Client reference |
| delivery_year | 2024, 2025 | Chronological filtering |
| type | use_case / summary | Chunk type distinction |

### Data Source

`Capability_Report.docx` — 10 use cases across AI & ML, Data Engineering, Cyber Security, and Digital Transformation, delivered 2024–2026 for Banking, Insurance, and Healthcare clients.

### External API

**Tavily Search API** — triggered when queries require latest market intelligence or internal results are insufficient. Returns structured results (URL, title, content) passed to the LLM alongside internal context.

---

## 4 — Prototype

### Components

| File | Purpose |
|------|---------|
| `src/ingest.py` | Document parsing, chunking, embedding, Qdrant upsert |
| `src/retriever.py` | Hybrid retrieval: metadata + semantic + BM25 |
| `src/retriever_hyde.py` | Advanced HyDE retriever |
| `src/agent.py` | LangGraph agent — 5 nodes, conditional routing |
| `src/evaluate.py` | RAGAS evaluation: baseline vs HyDE |
| `app.py` | Streamlit chat UI |

### Deployment

Deployed on **Render (free tier)**. API keys stored as environment variables in Render's dashboard — never committed to the repository.

---

## 5 — Baseline Evaluation

### Golden Dataset

13 question-answer pairs covering all capability domains and industries — designed to test single use case lookups, domain-filtered queries, count questions, and pitch preparation scenarios.

### RAGAS Results

| Question | Faithfulness | Context Precision | Context Recall | Answer Relevancy |
|----------|-------------|-------------------|----------------|-----------------|
| Banking use cases count? | 0.696 | 0.000 | 0.000 | 0.655 |
| 3 AI ML use cases delivered? | 0.912 | 1.000 | 0.750 | 0.638 |
| Latest Cyber Security use cases? | 0.571 | 1.000 | 0.500 | 0.706 |
| Data Engineering domains covered? | 0.545 | 0.000 | 1.000 | 0.740 |
| Digital Transformation clients & count? | 0.500 | 1.000 | 0.000 | 0.541 |
| Insurance sector experience? | 1.000 | 1.000 | 0.500 | 0.554 |
| CFO regulatory compliance use case? | 1.000 | 1.000 | 0.400 | 0.793 |
| Pitch fraud detection to prospect? | 0.567 | 1.000 | 0.000 | 0.498 |
| Healthcare client experience? | 0.842 | 1.000 | 0.500 | 0.619 |
| Retail bank CDO — which use case? | 0.839 | 0.000 | 0.000 | 0.862 |
| Total use cases & capability areas? | 0.792 | 1.000 | 0.200 | 0.818 |
| CISO security posture — financial firm? | 0.632 | 1.000 | 0.667 | 0.641 |
| Banks churn reduction experience? | 0.818 | 0.833 | 0.800 | 0.712 |

**Aggregate Scores:**

| Metric | Score |
|--------|-------|
| Faithfulness | 0.747 |
| Context Precision | 0.756 |
| Context Recall | 0.409 |
| Answer Relevancy | 0.675 |

### Conclusions

The pipeline is reasonably faithful (0.747) and precise (0.756) but suffers from poor context recall (0.409). Root cause: **vocabulary mismatch** — partners use business language (*"regulatory penalties"*) while documents use technical terms (*"Basel III", "BCBS 239"*). This motivated the HyDE upgrade in 6.

---

## 6 — Advanced Retriever Upgrade

### Technique: HyDE (Hypothetical Document Embeddings)

Instead of embedding the raw query, HyDE first asks the LLM to generate a hypothetical answer written in the style of the internal documents. This hypothetical answer is embedded and used as the search vector — bridging the vocabulary gap between business-language queries and technical-language documents.

**Example:** Query *"regulatory penalties and compliance delays"* → HyDE generates text containing *"Basel III, COREP, FINREP, BCBS 239"* → embedding lands much closer to the actual document vectors.

### Implementation (`src/retriever_hyde.py`)

1. Generate hypothetical answer via GPT-4o-mini
2. Embed hypothetical answer using `text-embedding-3-small`
3. Semantic search in Qdrant with metadata pre-filtering
4. BM25 re-ranking on the original query (preserves keyword matching)

### Results

| Metric | Baseline | HyDE | Change |
|--------|----------|------|--------|
| Faithfulness | 0.766 | 0.733 | ▼ -0.033 |
| Context Precision | 0.679 | 0.718 | ▲ +0.038 |
| Context Recall | 0.300 | 0.436 | ▲ +0.136 |
| Answer Relevancy | 0.628 | 0.681 | ▲ +0.053 |

### Analysis

HyDE improved 3 out of 4 metrics. The standout result is **context recall +0.136 (+45% relative)** — confirming vocabulary mismatch was the primary bottleneck. The minor faithfulness drop (-0.033) is an acceptable tradeoff given the overall quality improvement.

**Decision: HyDE retained as the production retrieval strategy.**

---

*KX — Nexvance Consulting Group (fictional) | AIM #09, AI Engineering Bootcamp 2026*
