import os
import re
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langsmith import traceable
from retriever import retrieve

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "KX-Certification-Challenge")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT

# ── Clients ───────────────────────────────────────────────────────────────────
llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
tavily_tool = TavilySearchResults(max_results=4, tavily_api_key=TAVILY_API_KEY)

# ── External intent keywords ──────────────────────────────────────────────────
EXTERNAL_INTENT_KEYWORDS = [
    "latest", "recent", "market", "industry trend", "2024", "2025", "2026",
    "globally", "worldwide", "competitors", "other firms", "what is happening",
    "emerging", "new use case", "best practice", "in the market", "out there",
]

# ── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query:            str
    in_scope:         bool
    internal_results: list
    web_results:      list
    need_web_search:  bool
    web_search_used:  bool
    final_answer:     str


# ── Node 0: LLM-based scope classification ───────────────────────────────────
def check_scope(state: AgentState) -> AgentState:
    print("\n[Agent] Step 0: Classifying query scope...")

    classification_prompt = """You are a query classifier for KX — an internal knowledge assistant for Nexvance Consulting Group.

KX can ONLY answer questions that are:
1. About use cases, projects, or engagements delivered BY Nexvance
2. About Nexvance's capabilities, clients, or domains (AI/ML, Data Engineering, Cyber Security, Digital Transformation)
3. Requests for market intelligence, latest trends, or external use cases to support Nexvance's sales pitches or client meetings — even if Nexvance hasn't delivered it themselves
4. Any query asking about "latest", "recent", or "market" use cases in domains where Nexvance operates (AI/ML, Cyber Security, Data Engineering, Digital Transformation, Banking, Insurance, Healthcare)

KX must REJECT questions that are:
- General industry knowledge not related to Nexvance's own work
- About other companies, public figures, or external organizations
- Political, personal, health, or lifestyle questions
- General "how to" questions unrelated to Nexvance's portfolio
- Questions about individuals (even if they mention AI or technology)

Query: "{query}"

Reply with ONE word only: INSCOPE or OUTOFSCOPE"""

    response = llm.invoke([
        HumanMessage(content=classification_prompt.format(query=state["query"]))
    ])

    result   = response.content.strip().upper()
    in_scope = "INSCOPE" in result
    print(f"[Agent] Scope classification: {result} → in_scope: {in_scope}")
    return {**state, "in_scope": in_scope}


# ── Conditional edge after scope check ───────────────────────────────────────
def route_after_scope(state: AgentState) -> str:
    return "retrieve_internal" if state["in_scope"] else "out_of_scope"


# ── Node: Out of scope response ───────────────────────────────────────────────
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
    results = retrieve(state["query"], top_k=4)
    print(f"[Agent] Retrieved {len(results)} internal chunks")
    return {**state, "internal_results": results}


# ── Node 2: Assess sufficiency with deterministic rules ──────────────────────
def assess_sufficiency(state: AgentState) -> AgentState:
    print("\n[Agent] Step 2: Assessing sufficiency of internal results...")
    query            = state["query"].lower()
    internal_results = state["internal_results"]

    if not internal_results:
        print("[Agent] No internal results — web search needed")
        return {**state, "need_web_search": True}

    if any(kw in query for kw in EXTERNAL_INTENT_KEYWORDS):
        print("[Agent] External intent detected — web search needed")
        return {**state, "need_web_search": True}

    number_match = re.search(r'\b(\d+)\b', query)
    if number_match:
        requested_n = int(number_match.group(1))
        if len(internal_results) < requested_n:
            print(f"[Agent] Requested {requested_n}, found {len(internal_results)} — web search needed")
            return {**state, "need_web_search": True}

    print("[Agent] Internal results sufficient — skipping web search")
    return {**state, "need_web_search": False}


# ── Conditional edge ──────────────────────────────────────────────────────────
def route_after_assessment(state: AgentState) -> str:
    return "web_search" if state["need_web_search"] else "generate_answer"


# ── Node 3: Web search via Tavily ─────────────────────────────────────────────
def web_search(state: AgentState) -> AgentState:
    print("\n[Agent] Step 3: Searching the web via Tavily...")
    results = tavily_tool.invoke({"query": state["query"]})
    print(f"[Agent] Retrieved {len(results)} web results")
    return {**state, "web_results": results, "web_search_used": True}


# ── Node 4: Generate final answer ────────────────────────────────────────────
def generate_answer(state: AgentState) -> AgentState:
    print("\n[Agent] Step 4: Generating final answer...")
    query            = state["query"]
    internal_results = state["internal_results"]
    web_results      = state.get("web_results", [])

    internal_context = ""
    if internal_results:
        internal_context = "--- INTERNAL KNOWLEDGE BASE ---\n" + "\n\n".join(
            f"[{r['use_case_id']}] {r['title']} (Client: {r['client']}, Domain: {r['capability_domain']}, Industry: {r['industry']}):\n{r['text']}"
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
- When a use case comes from web search, explicitly label it as "🌐 External Market Use Case (Web Source)"
- If combining internal and external sources, clearly distinguish them with separate sections"""

    user_prompt = f"""Partner Query: {query}

{internal_context}
{web_context}

Please provide a comprehensive, well-structured answer to the partner's query."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    # Append web sources if web search was used
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
    graph.add_conditional_edges(
        "check_scope",
        route_after_scope,
        {
            "retrieve_internal": "retrieve_internal",
            "out_of_scope":      "out_of_scope",
        }
    )
    graph.add_edge("out_of_scope",      END)
    graph.add_edge("retrieve_internal", "assess_sufficiency")
    graph.add_conditional_edges(
        "assess_sufficiency",
        route_after_assessment,
        {
            "web_search":      "web_search",
            "generate_answer": "generate_answer",
        }
    )
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
        "final_answer":     "",
    }
    final_state = kx_agent.invoke(initial_state)
    return final_state["final_answer"], final_state["web_search_used"]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What are the latest fraud detection use cases in the market in 2025?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        print('='*70)
        answer, web_used = run_agent(q)
        print(f"Web search used: {web_used}\n\nFINAL ANSWER:\n{answer}")
