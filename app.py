# app.py
# Purpose: Streamlit UI for KX — ChatGPT-style chat with fixed bottom input

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agent import run_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KX — Nexvance Knowledge Assistant",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0B0F1A;
    color: #E8E8E8;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 800px; }

/* ── Header ── */
.kx-header {
    text-align: center;
    padding: 1.2rem 0 1rem 0;
    border-bottom: 1px solid #1E2535;
    margin-bottom: 1.5rem;
}
.kx-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    letter-spacing: -2px;
    color: #C8A96E;
    line-height: 1;
}
.kx-tagline {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6B7A99;
    margin-top: 0.3rem;
}

/* ── Suggestion chips ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-bottom: 1.2rem;
    justify-content: center;
}
.chip {
    background: #141824;
    border: 1px solid #2A3550;
    color: #8899BB;
    padding: 0.3rem 0.8rem;
    border-radius: 16px;
    font-size: 0.72rem;
}

/* ── User bubble ── */
.user-bubble {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
}
.user-bubble-inner {
    background: #1E3A5F;
    color: #E8F0FF;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── KX bubble ── */
.kx-bubble {
    display: flex;
    justify-content: flex-start;
    gap: 0.6rem;
    align-items: flex-start;
    margin-bottom: 1.2rem;
}
.kx-avatar {
    background: #C8A96E;
    color: #0B0F1A;
    font-family: 'DM Serif Display', serif;
    font-size: 0.8rem;
    font-weight: bold;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.kx-bubble-inner {
    background: #FFFFFF;
    color: #1A1A2E;
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.3rem;
    max-width: 85%;
    font-size: 0.92rem;
    line-height: 1.75;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}
.kx-bubble-inner h1,.kx-bubble-inner h2,.kx-bubble-inner h3,.kx-bubble-inner h4 {
    color: #0B0F1A; margin-top: 0.8rem; margin-bottom: 0.3rem;
}
.kx-bubble-inner ul, .kx-bubble-inner ol { padding-left: 1.2rem; }
.kx-bubble-inner a { color: #1a73e8; word-break: break-all; }
.kx-bubble-inner table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.kx-bubble-inner th { background: #f0f0f0; padding: 6px 10px; border: 1px solid #ddd; }
.kx-bubble-inner td { padding: 6px 10px; border: 1px solid #ddd; }

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-bottom: 0.6rem;
}
.status-internal { background: #E8F5EE; color: #2E7D52; border: 1px solid #B5D9C5; }
.status-web      { background: #FFF8EC; color: #A07020; border: 1px solid #E8D090; }

/* ── Native chat input styling ── */
[data-testid="stChatInput"] {
    background: #141824 !important;
    border-top: 1px solid #1E2535 !important;
}
[data-testid="stChatInput"] textarea {
    min-height: 100px !important;
    max-height: 200px !important;
    background: #FFFFFF !important;
    color: #1A1A2E !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] button {
    background: #C8A96E !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kx-header">
    <div class="kx-logo">KX</div>
    <div class="kx-tagline">Nexvance · AI Powered Knowledge Assistant</div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggestion chips (only on fresh start) ───────────────────────────────────
SUGGESTIONS = [
    "AI & ML use cases in Banking",
    "Do we have Insurance experience?",
    "Latest Cyber Security use cases in market",
    "Pitch fraud detection to a prospect",
    "Total use cases we have delivered",
]

if not st.session_state.messages:
    st.markdown(
        '<div class="chip-row">' +
        "".join(f'<span class="chip">💬 {s}</span>' for s in SUGGESTIONS) +
        '</div>',
        unsafe_allow_html=True
    )

# ── Render chat history ───────────────────────────────────────────────────────
def render_markdown(text: str) -> str:
    try:
        import markdown
        return markdown.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        return text.replace("\n", "<br>")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-bubble">
            <div class="user-bubble-inner">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        badge = (
            '<span class="status-badge status-web">⚡ Internal + Web Search</span><br>'
            if msg.get("web_used") else
            '<span class="status-badge status-internal">✦ Internal Knowledge Base</span><br>'
        )
        html_content = render_markdown(msg["content"])
        st.markdown(f"""
        <div class="kx-bubble">
            <div class="kx-avatar">KX</div>
            <div class="kx-bubble-inner">{badge}{html_content}</div>
        </div>""", unsafe_allow_html=True)

# ── Fixed bottom chat input (native Streamlit — always stays at bottom) ───────
query = st.chat_input("Ask about Nexvance's use cases, capabilities, or market intelligence...")

if query and query.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query.strip()})

    with st.spinner("KX is thinking..."):
        answer, web_used = run_agent(query.strip())

    # Add KX response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "web_used": web_used,
    })

    st.rerun()
