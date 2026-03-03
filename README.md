# KX — AI-Powered Knowledge Assistant
### Nexvance Consulting Group | AIM Certification Challenge Submission

---

## 📋 Submission Checklist

| Item | Status |
|------|--------|
| GitHub Repo (public) | ✅ |
| Written Submission Document | ✅ [`KX_Submission_Document.docx`](./KX_Submission_Document.docx) |
| Live Demo (Render) | ✅ [KX Live App](#) *(replace with your Render URL)* |
| Loom Video (≤5 min) | ✅ [Watch Demo](#) *(replace with your Loom URL)* |

---

## 🧠 What is KX?

KX is an **Agentic RAG application** built for consulting firm partners and leadership to instantly surface:
- **Internal use cases** delivered by Nexvance Consulting Group
- **Latest market intelligence** from the web — when internal data isn't enough

Partners type natural language queries like *"Do we have fraud detection experience in Banking?"* and KX returns structured, referenced answers — sourced from internal knowledge first, web search only when needed.

---

## 🎯 Use Case

**Problem:** Consulting partners cannot quickly surface relevant past use cases before client meetings — creating team dependency and preparation delays.

**Solution:** Agentic RAG combining private internal knowledge retrieval (Qdrant) with real-time web search (Tavily), orchestrated by a LangGraph agent.

**User:** Partners, Directors, and Senior Managers at Nexvance Consulting Group.

---

## 🏗️ Architecture

```
Partner Query
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   LangGraph Agent                   │
│                                                     │
│  Node 0: LLM Scope Check (GPT-4o-mini)             │
│     │ OUTOFSCOPE → Reject with guidance             │
│     │ INSCOPE ↓                                     │
│  Node 1: Internal Retrieval (Qdrant)               │
│     │   ├─ Metadata Pre-filter                      │
│     │   ├─ HyDE Semantic Search                     │
│     │   └─ BM25 Re-ranking (RRF)                    │
│     ▼                                               │
│  Node 2: Sufficiency Assessment (Deterministic)    │
│     │ SUFFICIENT → Generate Answer                  │
│     │ INSUFFICIENT ↓                                │
│  Node 3: Web Search (Tavily API)                   │
│     ▼                                               │
│  Node 4: Answer Generation (GPT-4o-mini)           │
└─────────────────────────────────────────────────────┘
     │
     ▼
Structured Answer (Internal KB + 🌐 Web Sources)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | GPT-4o-mini (OpenAI) |
| Agent Orchestration | LangGraph |
| Web Search Tool | Tavily Search API |
| Embedding Model | text-embedding-3-small (OpenAI, 1536-dim) |
| Vector Database | Qdrant Cloud |
| Retriever | Hybrid: HyDE + Semantic + BM25 (RRF) |
| Monitoring | LangSmith |
| Evaluation | RAGAS |
| UI | Streamlit |
| Deployment | Render |

---

## 📁 Repository Structure

```
KX-Certification-Challenge/
│
├── src/
│   ├── ingest.py              # Doc parsing → chunking → embedding → Qdrant upsert
│   ├── retriever.py           # Hybrid retrieval: metadata + semantic + BM25
│   ├── retriever_hyde.py      # Advanced retriever: HyDE + hybrid
│   ├── agent.py               # LangGraph agent (5 nodes, conditional routing)
│   └── evaluate.py            # RAGAS evaluation: baseline vs HyDE comparison
│
├── data/
│   ├── Capability_Report.docx         # Internal knowledge base
│   ├── golden_dataset.csv             # 13 Q&A pairs for evaluation
│   ├── ragas_baseline_results.csv     # Baseline RAGAS scores
│   ├── ragas_hyde_results.csv         # HyDE RAGAS scores
│   ├── ragas_comparison.csv           # Side-by-side comparison
│   └── KX_Submission_Document.docx    # Full written submission (Tasks 1–7)
│
├── app.py                     # Streamlit chat UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Evaluation Results

### Baseline vs HyDE Retriever

| Metric | Baseline | HyDE | Change |
|--------|----------|------|--------|
| Faithfulness | 0.747 | 0.733 | ▼ -0.033 |
| Context Precision | 0.679 | 0.718 | ▲ +0.038 |
| Context Recall | 0.300 | 0.436 | ▲ +0.136 |
| Answer Relevancy | 0.628 | 0.681 | ▲ +0.053 |

**Key insight:** HyDE improved context recall by **+45% relative** (0.300 → 0.436) by bridging the vocabulary gap between partner queries (business language) and internal documents (technical terminology).

---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/KX-Certification-Challenge.git
cd KX-Certification-Challenge
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=KX-Certification-Challenge
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 5. Ingest data (first time only)
```bash
python src/ingest.py
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## 🧪 Running Evaluation

```bash
# Baseline + HyDE comparison
python src/evaluate.py
```

Results saved to `data/ragas_comparison.csv`.

---

## 📄 Submission Document

Full written responses to all 7 tasks are in [`KX_Submission_Document.docx`](./KX_Submission_Document.docx).

---

*AIM AI Engineering Bootcamp — Certification Challenge 2026*
