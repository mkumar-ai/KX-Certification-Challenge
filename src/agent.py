# agent.py
# Purpose: LangGraph agentic RAG — internal retrieval first, web search only if needed
# v5: Fixed year-as-count bug + show-all requests skip web search

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

# ── Year pattern — to distinguish years from counts ──────────────────────────
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')

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
    requested_n:      int   # actual count requested (0 = not specified)
    show_all:         bool  # True when user wants all available
    final_answer:     str


# ── Helper: extract requested count (ignoring years) ─────────────────────────
def extract_requested_count(query: str) -> tuple:
    """
    Returns (requested_n, show_all).
    - Strips year references before extracting count
    - show_all=True when requested_n > MAX_COLLECTION (typo like 10000)
    """
    # Remove year references so they don't get parsed as counts
    query_no_years = YEAR_PATTERN.sub("", query.lower())

    number_match = re.search(r'\b(\d+)\b', query_no_years)
    if not number_match:
        return 4, False  # default retrieval

    requested_n = int(number_match.group(1))

    # If number is unrealistically large → user wants "all" → don't web search
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
    top_k = max(requested_n, 4)

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

    # Rule 2: show_all=True means large/typo number → return all internal, no web search
    if show_all:
        print("[Agent] Large number detected (show all) — returning all internal, skipping web search")
        return {**state, "need_web_search": False}

    # Rule 3: Explicit external/market intent keywords
    if any(kw in query for kw in EXTERNAL_INTENT_KEYWORDS):
        print("[Agent] External intent detected — web search needed")
        return {**state, "need_web_search": True}

    # Rule 4: User asked for N items but found fewer internally
    if requested_n > 0 and len(internal_results) < requested_n:
        print(f"[Agent] Requested {requested_n}, found {len(internal_results)} — web search needed")
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
You help partners and leadership quickly surface relevant use cases and market intelligence.

Guidelines:
- Be concise, structured, and professional
- Always prioritise internal knowledge base results
- Use bullet points or tables when listing multiple use cases
- Include client names, outcomes, and key metrics when available
- When listing multiple use cases, always include the delivery year from metadata
- When asked to sort chronologically, sort by delivery_year ascending
- When a use case comes from web search, explicitly label it as "🌐 External Market Use Case (Web Source)"
- If combining internal and external sources, clearly distinguish them with separate sections
- Never fabricate metrics or outcomes not present in the provided context
- If a query references a year (e.g. 2025), filter and show only use cases delivered in that year"""

    user_prompt = f"""Partner Query: {query}

{internal_context}
{web_context}

Please provide a comprehensive, well-structured answer."""

    response = llm_best.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    answer = response.content
    if web_results:
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
        "How many use cases have been delivered in 2025?",
        "Show me top 10000 use cases",
        "Need 3 use cases related to cyber security",
        "Show me 3 use cases of AI ML and arrange them chronologically",
    ]
    for q in test_queries:
        print(f"\n{'='*70}\nQUERY: {q}\n{'='*70}")
        answer, web_used = run_agent(q)
        print(f"Web search used: {web_used}\n\nFINAL ANSWER:\n{answer}")
