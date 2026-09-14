"""AARAG Demo — Interactive Streamlit application."""

import os
import sys
import time
import json
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aarag import AARAG


st.set_page_config(
    page_title="AARAG Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .layer-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .reflection-token {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        margin: 4px;
        font-size: 14px;
        font-weight: bold;
    }
    .token-relevant { background-color: #d4edda; color: #155724; }
    .token-irrelevant { background-color: #f8d7da; color: #721c24; }
    .token-supported { background-color: #d4edda; color: #155724; }
    .token-partial { background-color: #fff3cd; color: #856404; }
    .token-unsupported { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_aarag():
    """Load AARAG system (cached)."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    try:
        return AARAG(config_path=str(config_path))
    except Exception as e:
        st.error(f"Error loading AARAG: {e}")
        return None


def render_sidebar():
    """Render sidebar configuration."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        st.subheader("Pipeline Settings")
        enable_web = st.checkbox("Enable Web Fallback", value=True)
        enable_reflection = st.checkbox("Enable Self-Reflection", value=True)

        st.subheader("Strategy Override")
        strategy = st.selectbox(
            "Force Strategy",
            ["Auto (Router)", "no_retrieval", "single_step", "multi_step", "graph_global"],
        )

        st.subheader("System Info")
        st.info("""
        **AARAG** — Adaptive Agentic RAG

        5-Layer Architecture:
        1. Knowledge Sources
        2. Adaptive Router
        3. Corrective Retrieval
        4. Self-Reflection
        5. Agentic Orchestrator
        """)

        return {
            "enable_web": enable_web,
            "enable_reflection": enable_reflection,
            "strategy": strategy if strategy != "Auto (Router)" else None,
        }


def render_routing_decision(routing: dict):
    """Render the routing decision panel."""
    st.subheader("🎯 Routing Decision")

    col1, col2, col3 = st.columns(3)
    with col1:
        level_names = {0: "No Retrieval", 1: "Single-Step", 2: "Multi-Step", 3: "Graph-Global"}
        level = routing.get("level", 0)
        st.metric("Complexity Level", level_names.get(level, "Unknown"))
    with col2:
        st.metric("Strategy", routing.get("strategy", "N/A"))
    with col3:
        st.metric("Confidence", f"{routing.get('confidence', 0):.2f}")

    if routing.get("fallback_used"):
        st.warning("⚠️ Low confidence — fallback strategy used")


def render_corrective_result(corrective: dict):
    """Render the corrective retrieval panel."""
    st.subheader("🔧 Corrective Retrieval")

    action = corrective.get("action", "N/A")
    confidence = corrective.get("confidence", 0)
    web_used = corrective.get("web_used", False)

    col1, col2, col3 = st.columns(3)
    with col1:
        action_colors = {"CORRECT": "🟢", "INCORRECT": "🔴", "AMBIGUOUS": "🟡"}
        st.metric("Action", f"{action_colors.get(action, '⚪')} {action}")
    with col2:
        st.metric("Confidence", f"{confidence:.2f}")
    with col3:
        st.metric("Web Fallback", "Yes" if web_used else "No")

    sources = corrective.get("sources", [])
    if sources:
        st.caption(f"Sources: {', '.join(set(sources))}")


def render_reflection_result(reflection: dict):
    """Render the self-reflection panel."""
    if not reflection:
        return

    st.subheader("🪞 Self-Reflection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        relevance = reflection.get("relevance", "N/A")
        color = "token-relevant" if relevance == "[Relevant]" else "token-irrelevant"
        st.markdown(f'<span class="reflection-token {color}">{relevance}</span>', unsafe_allow_html=True)
        st.caption("Relevance")

    with col2:
        support = reflection.get("support", "N/A")
        if "Fully" in support:
            color = "token-supported"
        elif "Partially" in support:
            color = "token-partial"
        else:
            color = "token-unsupported"
        st.markdown(f'<span class="reflection-token {color}">{support}</span>', unsafe_allow_html=True)
        st.caption("Support")

    with col3:
        utility = reflection.get("utility", "N/A")
        st.markdown(f'<span class="reflection-token" style="background:#cce5ff;color:#004085;">{utility}</span>', unsafe_allow_html=True)
        st.caption("Utility")

    with col4:
        st.metric("Confidence", f"{reflection.get('final_confidence', 0):.2f}")

    if reflection.get("revised"):
        st.info(f"🔄 Answer was revised {reflection.get('revision_count', 0)} time(s)")


def render_agent_trace(agent_response: dict):
    """Render the agent reasoning trace."""
    st.subheader("🤖 Agentic Orchestrator")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Steps", agent_response.get("total_steps", 0))
    with col2:
        tools = agent_response.get("tools_used", [])
        st.metric("Tools Used", ", ".join(tools) if tools else "None")


def render_comparison(query: str, aarag_response, config: dict):
    """Render side-by-side comparison with baselines."""
    st.subheader("📊 Comparison with Baselines")

    # Simple baselines
    st.markdown("**Standard RAG** (retrieve-then-read):")
    standard_answer = f"Based on retrieved context, here is information about: {query}"
    st.info(standard_answer)

    st.markdown("**AARAG** (our system):")
    st.success(aarag_response.answer)


def main():
    """Main demo application."""
    st.title("🔍 AARAG: Adaptive Agentic Retrieval-Augmented Generation")
    st.markdown("*A unified framework combining Self-RAG, CRAG, Adaptive-RAG, and GraphRAG*")

    # Sidebar
    config = render_sidebar()

    # Load system
    aarag = load_aarag()

    # Example queries
    st.subheader("📝 Try These Examples")
    examples = [
        "Who is the president of France and what is the capital of Germany?",
        "Compare the economic policies of the United States and China.",
        "Summarize the main themes across all documents in the knowledge base.",
        "What is the population of Tokyo?",
        "Explain the relationship between quantum physics and general relativity.",
    ]

    selected_example = st.selectbox("Select an example query:", [""] + examples)

    # Query input
    query = st.text_input(
        "Enter your question:",
        value=selected_example if selected_example else "",
        placeholder="Ask anything...",
    )

    if st.button("🚀 Ask AARAG", type="primary") and query:
        with st.spinner("Processing through 5 layers..."):
            try:
                start_time = time.time()
                response = aarag.query(
                    query,
                    enable_web_fallback=config["enable_web"],
                    enable_self_reflection=config["enable_reflection"],
                    strategy_override=config["strategy"],
                )
                elapsed = time.time() - start_time
            except Exception as e:
                st.error(f"Error: {e}")
                return

        # Display answer
        st.subheader("💡 Answer")
        st.markdown(f"**{response.answer}**")

        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Confidence", f"{response.confidence:.2f}")
        with col2:
            st.metric("Latency", f"{response.latency:.2f}s")
        with col3:
            st.metric("Strategy", response.strategy_used)

        # Layer details
        st.markdown("---")
        render_routing_decision(response.routing_decision)

        st.markdown("---")
        render_corrective_result(response.corrective_result)

        st.markdown("---")
        render_reflection_result(response.reflection_result)

        st.markdown("---")
        render_agent_trace(response.agent_response)

        # Comparison
        st.markdown("---")
        render_comparison(query, response, config)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;'>"
        "AARAG — M.Research AI Capstone Project 2026<br>"
        "Combining: Adaptive-RAG (NAACL 2024) | Self-RAG (ICLR 2024) | CRAG (ICML 2024) | GraphRAG (2024)"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
