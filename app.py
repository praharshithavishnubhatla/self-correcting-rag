"""
app.py — Streamlit UI for Self-Correcting RAG
Run with:  streamlit run app.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Self-Correcting RAG",
    page_icon="🔍",
    layout="centered",
)

st.markdown("""
<style>
.answer-box {
    background: #f0f4ff;
    border-left: 4px solid #4f8ef7;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    font-size: 1rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.source-tag {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 4px;
    padding: 2px 10px;
    margin: 3px 2px;
    font-size: 0.82rem;
    font-family: monospace;
}
.fail-box {
    background: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 0.8rem 1rem;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader (cached) ──────────────────────────────

@st.cache_resource(show_spinner="Loading pipeline and indexes…")
def load_pipeline():
    try:
        from rag.rag import run_pipeline
        return run_pipeline
    except Exception as e:
        return None


run_pipeline = load_pipeline()


# ── Sidebar ───────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    debug_mode = st.toggle("Debug mode", value=False)

    st.divider()
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        st.success("✅ GROQ_API_KEY loaded")
    else:
        st.error("❌ GROQ_API_KEY missing")
        manual_key = st.text_input("Enter Groq API key", type="password")
        if manual_key:
            os.environ["GROQ_API_KEY"] = manual_key
            st.success("Key set for this session")

    st.divider()
    st.caption("**Knowledge base**")
    st.markdown("""
- `design.md` — URL Shortener design
- `design.txt` — Scaling, caching, CDN
- `system.md`  — Scalable system design
- `system.txt` — System design basics
""")

    st.divider()
    st.caption("[View on GitHub](https://github.com/praharshithavishnubhatla/self-correcting-rag)")


# ── Header ────────────────────────────────────────────────

st.title("🔍 Self-Correcting RAG")
st.caption(
    "FAISS + BM25 hybrid retrieval · RRF fusion · Query rewriting · "
    "Cross-encoder reranking · Guardrail + Evaluator agents"
)
st.divider()

# ── Example questions from your actual docs ───────────────
EXAMPLES = [
    "what is scalability?",
    "how does a load balancer work?",
    "explain caching strategies",
    "what is the CAP theorem?",
    "how would you design a URL shortener?",
]

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
selected_example = None
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        selected_example = ex

# ── Query input ───────────────────────────────────────────

query = st.text_input(
    "Ask a question",
    value=selected_example or "",
    placeholder="e.g. explain horizontal vs vertical scaling",
    label_visibility="collapsed",
)

ask_btn = st.button("Ask", type="primary")


# ── Run pipeline ──────────────────────────────────────────

if ask_btn and query.strip():
    if run_pipeline is None:
        st.error(
            "Pipeline failed to load. Make sure you've run `python main.py` "
            "to build the indexes first."
        )
    elif not os.getenv("GROQ_API_KEY"):
        st.warning("Please add your GROQ_API_KEY in the sidebar or .env file.")
    else:
        with st.spinner("Retrieving · Reranking · Generating…"):
            start = time.time()
            try:
                result = run_pipeline(query.strip(), debug=debug_mode)
                elapsed = round(time.time() - start, 2)

                answer = result.get("answer", "")
                sources = result.get("sources", [])
                guardrail_ok = result.get("guardrail_passed", False)
                verdict = result.get("evaluator_verdict", "—")

                # ── Answer ────────────────────────────────
                st.subheader("Answer")
                if guardrail_ok:
                    st.markdown(
                        f"<div class='answer-box'>{answer}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='fail-box'>{answer}</div>",
                        unsafe_allow_html=True
                    )

                # ── Sources ───────────────────────────────
                if sources:
                    st.markdown("<br>**Sources**", unsafe_allow_html=True)
                    html = " ".join(f"<span class='source-tag'>{s}</span>" for s in sources)
                    st.markdown(html, unsafe_allow_html=True)

                # ── Pipeline metrics ──────────────────────
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("⏱ Latency", f"{elapsed}s")
                c2.metric("🛡 Guardrail", "PASS" if guardrail_ok else "FAIL")
                c3.metric("✅ Evaluator", verdict)
                c4.metric("📄 Sources", len(sources))

                # ── Debug expander ────────────────────────
                if debug_mode:
                    with st.expander("🔬 Debug info"):
                        st.json(result.get("debug_info", {}))

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

elif ask_btn:
    st.warning("Please type a question first.")


# ── Query history ─────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if ask_btn and query.strip():
    if query.strip() not in st.session_state.history:
        st.session_state.history.append(query.strip())

if st.session_state.history:
    with st.expander(f"🕘 Query history ({len(st.session_state.history)})"):
        for q in reversed(st.session_state.history[-15:]):
            st.markdown(f"- {q}")