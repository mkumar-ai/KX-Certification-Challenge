import os
import re
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langsmith import traceable
from retriever import retrieve, MAX_COLLECTION

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "KX-Certification-Challenge")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT

# ── Clients ───────────────────────────────────────────────────────────────────
llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
llm_best = ChatOpenAI(model="gpt-4o",      temperature=0, api_key=OPENAI_API_KEY)

tavily_tool = TavilySearchResults(max_results=4, tavily_api_key=TAVILY_API_KEY)

# ── Year pattern — to distinguish years from counts ───────────────────────────
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')

# ── Show-all keywords — retrieve full collection ──────────────────────────────
SHOW_ALL_KEYWORDS = [
    "all", "total", "every", "breakdown", "split",
    "by industry", "by domain", "across", "each industry",
    "each domain", "full list", "complete list", "how many",
    "count", "summary", "overview", "all use cases",
]

# ── External intent keywords ──────────────────────────────────────────────────
EXTERNAL_INTENT_KEYWORDS = [
    "latest", "recent", "market", "industry trend",
    "globally", "worldwide", "competitors", "other firms",
    "what is happening", "emerging", "best practice",
    "in the market", "out there", "across the industry",
]

# ── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query:            str
    in_scope:         bool
    internal_results: list
    web_results:      list
    need_web_search:  bool
    web_search_used:  bool
    requested_n:      int   # 0 = not specified by user
    show_all:         bool
    final_answer:     str


# ── Helper: extract requested count (ignoring years) ─────────────────────────
def extract_requested_count(query: str) -> tuple:
    """
    Returns (requested_n, show_all).
    - Strips year references before extracting count
    - requested_n=0 means user did NOT specify a number
    - show_all=True when show-all keywords detected OR requested_n > MAX_COLLECTION
    """
    q_lower = query.lower()

    # Check for show-all intent keywords first
    if any(kw in q_lower for kw in SHOW_ALL_KEYWORDS):
        return MAX_COLLECTION, True

    # Remove year references so they don't get parsed as counts
    query_no_years = YEAR_PATTERN.sub("", q_lower)

    number_match = re.search(r'\b(\d+)\b', query_no_years)
    if not number_match:
        return 0, False  # no number specified — do NOT default to 4

    requested_n = int(number_match.group(1))

    # Unrealistically large number → treat as show all
    if requested_n > MAX_COLLECTION:
        return MAX_COLLECTION, True

    return requested_n, False


# ── Node 0: LLM-based scope classification ───────────────────────────────────
def check_scope(state: AgentState) -> AgentState:
    print("\n[Agent] Step 0: Classifying query scope...")

    classification_prompt = """You are a query classifier for KX, an internal knowledge assistant for Nexvance Consulting Group — a consulting firm specialising in AI/ML, Data Engineering, Cyber Security, and Digital Transformation for Banking, Insurance, and Healthcare clients.

Your job is to decide if a query is relevant to a consulting knowledge assistant.

INSCOPE — answer YES to any of these:
- Asks about use cases, projects, engagements, capabilities, or experience (in ANY phrasing)
- Asks about industries, domains, or clients a consulting firm might serve
- Asks for examples, references, or case studies to support a pitch or meeting
- Asks about market trends, latest developments, or benchmarks in technology or industry
- Could reasonably come from a consultant, partner, or business leader preparing for client work

OUTOFSCOPE — only reject if clearly:
- Personal lifestyle questions (health, fitness, relationships, hobbies)
- General factual questions about public figures, politics, or world events unrelated to business/technology
- Requests completely unrelated to consulting, technology, or business

When in doubt — classify as INSCOPE.

Query: "{query}"

Reply with ONE word only: INSCOPE or OUTOFSCOPE"""

    response = llm_fast.invoke([
        HumanMessage(content=classification_prompt.format(query=state["query"]))
    ])
    result   = response.content.strip().upper()
    in_scope = "INSCOPE" in result
    print(f"[Agent] Scope: {result} → in_scope: {in_scope}")
    return {**state, "in_scope": in_scope}


def route_after_scope(state: AgentState) -> str:
    return "retrieve_internal" if state["in_scope"] else "out_of_scope"


def out_of_scope(state: AgentState) -> AgentState:
    answer = (
        "I'm KX — Nexvance's consulting knowledge assistant. "
        "I can only help with questions related to our delivered use cases, "
        "capability areas, client engagements, and market intelligence.\n\n"
        "Try asking something like:\n"
        "- *What AI & ML use cases have we delivered in Banking?*\n"
        "- *Do we have experience in the Insurance sector?*\n"
        "- *Latest Cyber Security use cases in the market for our pitch?*"
    )
    return {**state, "final_answer": answer, "web_search_used": False}


# ── Node 1: Retrieve from internal knowledge base ────────────────────────────
def retrieve_internal(state: AgentState) -> AgentState:
    print("\n[Agent] Step 1: Searching internal knowledge base...")

    requested_n, show_all = extract_requested_count(state["query"])

    # Determine how many chunks to retrieve
    if show_all:
        top_k = MAX_COLLECTION
    elif requested_n > 0:
        top_k = max(requested_n, 4)
    else:
        top_k = 4  # sensible default when no number specified

    results = retrieve(state["query"], top_k=top_k)
    print(f"[Agent] Retrieved {len(results)} chunks | requested_n={requested_n} | show_all={show_all}")
    return {**state, "internal_results": results, "requested_n": requested_n, "show_all": show_all}


# ── Node 2: Assess sufficiency ────────────────────────────────────────────────
def assess_sufficiency(state: AgentState) -> AgentState:
    print("\n[Agent] Step 2: Assessing sufficiency...")
    query            = state["query"].lower()
    internal_results = state["internal_results"]
    requested_n      = state["requested_n"]
    show_all         = state["show_all"]

    # Rule 1: No internal results at all
    if not internal_results:
        print("[Agent] No internal results — web search needed")
        return {**state, "need_web_search": True}

    # Rule 2: show_all → return all internal, never web search
    if show_all:
        print("[Agent] Show-all intent — returning all internal, skipping web search")
        return {**state, "need_web_search": False}

    # Rule 3: Explicit external/market intent keywords
    if any(kw in query for kw in EXTERNAL_INTENT_KEYWORDS):
        print("[Agent] External intent detected — web search needed")
        return {**state, "need_web_search": True}

    # Rule 4: User EXPLICITLY asked for N items and we found fewer
    # Only applies when user specified a number AND it's within collection size
    if requested_n > 0 and requested_n <= MAX_COLLECTION and len(internal_results) < requested_n:
        print(f"[Agent] User explicitly requested {requested_n}, found {len(internal_results)} — web search needed")
        return {**state, "need_web_search": True}

    print("[Agent] Internal results sufficient — skipping web search")
    return {**state, "need_web_search": False}


def route_after_assessment(state: AgentState) -> str:
    return "web_search" if state["need_web_search"] else "generate_answer"


# ── Node 3: Web search via Tavily ─────────────────────────────────────────────
def web_search(state: AgentState) -> AgentState:
    print("\n[Agent] Step 3: Searching the web via Tavily...")
    results = tavily_tool.invoke({"query": state["query"]})
    print(f"[Agent] Retrieved {len(results)} web results")
    return {**state, "web_results": results, "web_search_used": True}


# ── Node 4: Generate final answer (GPT-4o) ───────────────────────────────────
def generate_answer(state: AgentState) -> AgentState:
    print("\n[Agent] Step 4: Generating final answer (GPT-4o)...")
    query            = state["query"]
    internal_results = state["internal_results"]
    web_results      = state.get("web_results", [])

    internal_context = ""
    if internal_results:
        internal_context = "--- INTERNAL KNOWLEDGE BASE ---\n" + "\n\n".join(
            f"[{r['use_case_id']}] {r['title']}\n"
            f"Client: {r['client']} | Domain: {r['capability_domain']} | "
            f"Industry: {r['industry']} | Delivered: {r.get('delivery_year', 'N/A')}\n"
            f"{r['text']}"
            for r in internal_results
        )

    web_context = ""
    if web_results:
        web_context = "\n\n--- EXTERNAL WEB SEARCH RESULTS ---\n" + "\n\n".join(
            f"[WEB SOURCE] {r.get('url', 'N/A')}\n{r.get('content', '')[:600]}"
            for r in web_results
        )

    system_prompt = """You are KX — an intelligent knowledge assistant for Nexvance Consulting Group.
You help partners quickly surface relevant use cases and market intelligence before client meetings.

## RESPONSE FORMAT RULES — FOLLOW STRICTLY

### Rule 1: Detect query intent first

- SUMMARY intent ("how many", "total", "count", "breakdown", "split by", "by industry", "by domain", "overview"):
  Respond with a brief structured summary ONLY. Show counts grouped by domain/industry.
  Do NOT show full use case details. Keep to 8-10 lines max.

- DETAIL intent ("show me", "tell me about", "describe", "what did we deliver", "explain"):
  Show full use case details using the format in Rule 3.

- PITCH intent ("pitch", "prospect", "meeting with", "lead with", "strongest outcome"):
  Show the 1-2 most relevant use cases with emphasis on measurable outcomes and metrics.

### Rule 2: Summary response format
CRITICAL: Build your response EXCLUSIVELY from the use cases provided in the context below.
Do NOT reference any industry, domain, or client not explicitly present in the retrieved chunks.
If an industry has zero use cases in the retrieved context — do NOT mention it at all. Not even to say "no use cases found".

Example structure:
**We have delivered X use cases:**

**[Industry 1]** — X use cases
- Use Case Title (Client, Year)

**[Industry 2]** — X use cases  
- Use Case Title (Client, Year)

Only include industries/domains that have actual use cases in the retrieved context.
NEVER mention an industry or domain with zero use cases — simply omit it.

### Rule 3: Detail response format
For EACH use case use this EXACT format. Always put a divider (---) between use cases:

---
**[UC#] Use Case Title**
| Field | Value |
|-------|-------|
| Client | Name |
| Domain | Domain |
| Industry | Industry |
| Delivered | Year |

**Challenge:** One concise sentence.
**Solution:** One concise sentence.
**Key Outcomes:**
- Outcome 1 with metric
- Outcome 2 with metric
- Outcome 3 with metric
---

### Rule 4: Web sources
- ONLY append a web sources section if external web results were actually provided and used
- NEVER show web source links for purely internal responses
- When combining internal + web, use clear section headings to separate them

### Rule 5: General rules
- Always prioritise internal knowledge base results
- NEVER fabricate metrics, client names, or outcomes not present in the provided context
- NEVER mention industries, domains, or clients not present in the retrieved chunks
- When sorting chronologically, sort by delivery_year ascending
- Be concise and professional — partners are time-pressed"""

    user_prompt = f"""Partner Query: {query}

{internal_context}
{web_context}

Please provide a well-structured answer following the format rules above."""

    response = llm_best.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    answer = response.content

    # Only append web sources if web search was actually used
    if web_results and state.get("web_search_used"):
        sources = "\n\n---\n**🌐 Web Sources:**\n" + "\n".join(
            f"- [{r.get('url', 'N/A')}]({r.get('url', 'N/A')})"
            for r in web_results if r.get("url")
        )
        answer += sources

    return {**state, "final_answer": answer}


# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("check_scope",        check_scope)
    graph.add_node("out_of_scope",       out_of_scope)
    graph.add_node("retrieve_internal",  retrieve_internal)
    graph.add_node("assess_sufficiency", assess_sufficiency)
    graph.add_node("web_search",         web_search)
    graph.add_node("generate_answer",    generate_answer)

    graph.set_entry_point("check_scope")
    graph.add_conditional_edges("check_scope", route_after_scope, {
        "retrieve_internal": "retrieve_internal",
        "out_of_scope":      "out_of_scope",
    })
    graph.add_edge("out_of_scope",      END)
    graph.add_edge("retrieve_internal", "assess_sufficiency")
    graph.add_conditional_edges("assess_sufficiency", route_after_assessment, {
        "web_search":      "web_search",
        "generate_answer": "generate_answer",
    })
    graph.add_edge("web_search",      "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────
kx_agent = build_graph()

@traceable(name="KX Agent Run")
def run_agent(query: str) -> tuple:
    """Returns: (final_answer, web_search_used)"""
    initial_state: AgentState = {
        "query":            query,
        "in_scope":         True,
        "internal_results": [],
        "web_results":      [],
        "need_web_search":  False,
        "web_search_used":  False,
        "requested_n":      0,
        "show_all":         False,
        "final_answer":     "",
    }
    final_state = kx_agent.invoke(initial_state)
    return final_state["final_answer"], final_state["web_search_used"]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "SHow many use cases we delivered in Data Engineering in Banking industry",
    ]
    for q in test_queries:
        print(f"\n{'='*70}\nQUERY: {q}\n{'='*70}")
        answer, web_used = run_agent(q)
        print(f"Web search used: {web_used}\n\nFINAL ANSWER:\n{answer}")
