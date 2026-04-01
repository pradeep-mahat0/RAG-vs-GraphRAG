"""
GraphRAG - Streamlit Chat Interface
Run: streamlit run app.py

Modes:
  - Local:  entity-graph-expanded retrieval (best for specific questions)
  - Global: community-summary-based retrieval (best for thematic/broad questions)
  - Hybrid: combines both
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🕸️ GraphRAG",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .chunk-box {
        background: #f0fdf4;
        border-left: 3px solid #11998e;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #444;
        margin-bottom: 0.5rem;
    }
    .summary-box {
        background: #f0f4ff;
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .timestamp-badge {
        background: #11998e22;
        color: #11998e;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .source-badge {
        background: #6366f122;
        color: #6366f1;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .entity-tag {
        display: inline-block;
        background: #11998e;
        color: white;
        border-radius: 8px;
        padding: 1px 8px;
        font-size: 0.72rem;
        margin: 2px;
    }
    .mode-local  { background: #11998e22; color: #11998e; border-radius: 8px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
    .mode-global { background: #6366f122; color: #6366f1; border-radius: 8px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
    .mode-hybrid { background: #f5922122; color: #f59221; border-radius: 8px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
    .stat-box {
        background: linear-gradient(135deg, #11998e22, #38ef7d22);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #11998e33;
    }
    .stat-number { font-size: 1.6rem; font-weight: 700; color: #11998e; }
    .stat-label  { font-size: 0.78rem; color: #888; }
    .community-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages":       [],
        "rag":            None,
        "ingested":       False,
        "ingest_stats_list": [],
        "active_sources": [],
        "query_mode":     "hybrid",
        "api_key":        os.getenv("OPENAI_API_KEY", "") or os.getenv("GROQ_API_KEY", ""),
        "provider":       "openai",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── LLM helpers ──────────────────────────────────────────────────────────────

def make_llm_fn(provider: str, api_key: str):
    """Return a simple llm_fn(prompt) -> str for use during ingestion (summaries)."""
    def llm_fn(prompt: str) -> str:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            resp   = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.3,
            )
            return resp.choices[0].message.content
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp   = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.3,
            )
            return resp.choices[0].message.content
    return llm_fn


def call_llm(messages: list, context: str, mode: str, provider: str, api_key: str) -> str:
    active_sources = st.session_state.get("active_sources", [])
    source_list    = ", ".join(active_sources) if active_sources else "unknown"

    mode_instruction = {
        "local":          "You have been given entity-graph-expanded excerpts most relevant to the question.",
        "global":         "You have been given thematic community summaries and associated excerpts. Focus on synthesizing themes.",
        "hybrid":         "You have been given both entity-specific excerpts and thematic community summaries. Combine both for a comprehensive answer.",
        "global_fallback":"Community summaries are not available yet; excerpts were retrieved via standard retrieval.",
    }.get(mode, "")

    system_prompt = f"""You are an intelligent assistant analyzing podcast transcripts using a Knowledge Graph enhanced RAG system.
Sources loaded: {source_list}

{mode_instruction}

RULES:
1. Answer ONLY using information from the provided context below.
2. If the context doesn't contain enough information, say: "This topic isn't sufficiently covered in the available transcripts."
3. Cite which source/transcript and timestamp information comes from (e.g., "In [source] around 12:34, ...").
4. For global/thematic questions, synthesize across the provided topic summaries.
5. Keep answers concise and grounded. Do NOT guess or use outside knowledge.

RETRIEVED CONTEXT (Knowledge Graph enhanced):
{context}"""

    full_messages = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
    full_messages.append({"role": "user", "content": messages[-1]["content"]})

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}] + full_messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp   = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + full_messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return resp.choices[0].message.content


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.session_state.provider = st.selectbox(
        "LLM Provider",
        ["openai", "groq"],
        index=0 if st.session_state.provider == "openai" else 1,
    )

    api_key_label = "OpenAI API Key" if st.session_state.provider == "openai" else "Groq API Key"
    api_key_input = st.text_input(
        api_key_label, value=st.session_state.api_key,
        type="password", placeholder="Enter your API key..."
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.divider()

    with st.expander("🔧 RAG Settings", expanded=False):
        chunk_size = st.slider("Chunk Size (tokens)", 200, 800, 500, 50)
        overlap    = st.slider("Overlap (tokens)",     20, 150,  80, 10)
        top_k      = st.slider("Retrieve Top-K",        5,  30,  20,  5)
        top_n      = st.slider("Rerank to Top-N",        2,  10,   5,  1)

    st.divider()

    # ── File Upload ────────────────────────────────────────────────────────────
    st.markdown("## 📂 Upload Transcripts")
    uploaded_files = st.file_uploader(
        "Upload .srt / .txt files",
        type=["srt", "txt"],
        accept_multiple_files=True,
        help="Multiple transcripts supported — entities will be linked across all of them."
    )

    generate_summaries_on_ingest = st.checkbox(
        "Generate community summaries (uses LLM)",
        value=True,
        help="If checked, LLM summaries are generated for each community cluster during ingestion. Required for Global query mode."
    )

    if uploaded_files:
        already  = st.session_state.active_sources or []
        pending  = [f for f in uploaded_files if Path(f.name).stem not in already]

        btn_label = (
            f"🚀 Process {len(pending)} Transcript{'s' if len(pending) != 1 else ''}"
            if pending else "✅ All files already loaded"
        )

        if st.button(btn_label, use_container_width=True, type="primary", disabled=not pending):
            if st.session_state.rag is None:
                from graph_pipeline import GraphRAG
                st.session_state.rag = GraphRAG(
                    persist_dir="./graph_store",
                    chunk_size=chunk_size,
                    overlap=overlap,
                    top_k_retrieve=top_k,
                    top_n_rerank=top_n
                )

            llm_fn = None
            if generate_summaries_on_ingest and st.session_state.api_key:
                llm_fn = make_llm_fn(st.session_state.provider, st.session_state.api_key)

            progress = st.progress(0, text="Starting…")
            for i, uf in enumerate(pending):
                progress.progress(i / len(pending), text=f"Processing: {uf.name}")
                tmp_path = Path(f"/tmp/{uf.name}")
                tmp_path.write_bytes(uf.getvalue())
                try:
                    stats = st.session_state.rag.ingest(str(tmp_path), llm_fn=llm_fn)
                    st.session_state.ingest_stats_list.append(stats)
                except Exception as e:
                    st.error(f"❌ Error processing {uf.name}: {e}")
                    st.exception(e)

            progress.progress(1.0, text="Done!")
            st.session_state.ingested       = True
            st.session_state.active_sources = list(st.session_state.rag._ingested_sources.keys())
            st.success(f"✅ {len(pending)} transcript(s) ready!")
            st.rerun()

    # ── Generate Summaries Button (if not done during ingest) ─────────────────
    if st.session_state.ingested and st.session_state.rag is not None:
        rag = st.session_state.rag
        if not rag.has_summaries():
            st.divider()
            st.warning("⚠️ No community summaries yet. Global mode will fall back to local retrieval.")
            if st.button("📝 Generate Summaries Now", use_container_width=True):
                if not st.session_state.api_key:
                    st.error("API key required to generate summaries.")
                else:
                    with st.spinner("Generating community summaries with LLM..."):
                        llm_fn = make_llm_fn(st.session_state.provider, st.session_state.api_key)
                        rag.generate_summaries(llm_fn)
                    st.success("✅ Summaries generated!")
                    st.rerun()

    # ── Active Sources ─────────────────────────────────────────────────────────
    if st.session_state.active_sources:
        st.divider()
        st.markdown("## 📚 Active Transcripts")
        for src in st.session_state.active_sources:
            st.markdown(f'<span style="background:#11998e;color:white;border-radius:8px;padding:2px 8px;font-size:0.75rem;margin-right:4px">🎙 {src}</span>', unsafe_allow_html=True)

    # ── Graph Stats ────────────────────────────────────────────────────────────
    if st.session_state.ingested and st.session_state.rag:
        st.divider()
        st.markdown("## 📊 Graph Stats")
        gs   = st.session_state.rag.get_graph_stats()
        cols = st.columns(2)
        for (label, val), col in zip(
            [("Entities", gs["nodes"]), ("Edges", gs["edges"]),
             ("Communities", gs["communities"]), ("Density", gs["density"])],
            cols * 2
        ):
            col.markdown(f"""<div class="stat-box">
                <div class="stat-number">{val}</div>
                <div class="stat-label">{label}</div>
            </div><br>""", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("♻️ Reset Everything", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─── Main UI ──────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🕸️ GraphRAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Knowledge Graph Enhanced RAG · '
    'Entity Expansion + Community Detection + Hybrid Query Modes</div>',
    unsafe_allow_html=True
)

if not st.session_state.ingested:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **📥 Step 1: Upload**
        Upload `.srt` / `.txt` transcripts in the sidebar.
        """)
    with col2:
        st.markdown("""
        **🕸️ Step 2: Graph Build**
        Entities are extracted and linked into a knowledge graph.
        """)
    with col3:
        st.markdown("""
        **🔍 Step 3: Communities**
        Related entities are clustered and summarized by the LLM.
        """)
    with col4:
        st.markdown("""
        **💬 Step 4: Query**
        Choose Local / Global / Hybrid mode to chat.
        """)
    st.markdown("---")
    st.info("👈 Upload transcripts in the sidebar to get started.")

else:
    rag     = st.session_state.rag
    sources = st.session_state.active_sources
    src_str = " · ".join(f"`{s}`" for s in sources)
    st.markdown(f"**📄 Active ({len(sources)} transcript{'s' if len(sources) != 1 else ''}):** {src_str}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸️ Knowledge Graph"])

    # ── Chat Tab ───────────────────────────────────────────────────────────────
    with tab_chat:
        # Query mode selector
        st.session_state.query_mode = st.radio(
            "Query Mode",
            ["hybrid", "local", "global"],
            horizontal=True,
            format_func=lambda m: {
                "local":  "🎯 Local (entity-expanded)",
                "global": "🌐 Global (community themes)",
                "hybrid": "⚡ Hybrid (best of both)"
            }[m],
            help=(
                "**Local**: Graph-expanded entity retrieval — best for specific factual questions.\n\n"
                "**Global**: Uses community summaries — best for thematic/broad questions.\n\n"
                "**Hybrid**: Combines both approaches."
            )
        )
        st.markdown("---")

        # Render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    if msg.get("mode"):
                        mode_css = f"mode-{msg['mode'].split('_')[0]}"
                        st.markdown(f'<span class="{mode_css}">Mode: {msg["mode"]}</span>', unsafe_allow_html=True)

                    if msg.get("entities"):
                        entity_tags = " ".join(f'<span class="entity-tag">{e}</span>' for e in msg["entities"][:15])
                        st.markdown(f"**Entities used:** {entity_tags}", unsafe_allow_html=True)

                    if msg.get("summaries"):
                        with st.expander("🌐 Community Summaries Used", expanded=False):
                            for s in msg["summaries"]:
                                st.markdown(f'<div class="summary-box">{s}</div>', unsafe_allow_html=True)

                    if msg.get("hits"):
                        with st.expander(f"📎 View {len(msg['hits'])} Source Excerpts", expanded=False):
                            for j, (text, meta, score) in enumerate(msg["hits"], 1):
                                src = meta.get("source", "unknown")
                                ts  = f"{meta.get('start_time', '?')} → {meta.get('end_time', '?')}"
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'<span class="source-badge">📂 {src}</span> '
                                    f'<span class="timestamp-badge">⏱ {ts}</span> '
                                    f'<span style="color:#aaa;font-size:0.75rem"> · Score: {score:.3f}</span>'
                                    f'<br><br>{text}</div>',
                                    unsafe_allow_html=True
                                )

        # Suggestions
        if not st.session_state.messages:
            st.markdown("**💡 Try asking:**")
            suggestions = [
                "What are the main themes across all transcripts?",
                "Who are the key people mentioned and what did they say?",
                "What topics do the episodes have in common?",
                "Give me a high-level overview of the content.",
            ]
            cols = st.columns(2)
            for i, sug in enumerate(suggestions):
                if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state._prefill = sug
                    st.rerun()

        # Chat input
        user_input = st.chat_input("Ask anything about the transcripts...")

        if user_input:
            if not st.session_state.api_key:
                st.error("⚠️ Please enter your API key in the sidebar.")
                st.stop()

            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                mode = st.session_state.query_mode
                with st.spinner(f"🔍 Running {mode} graph query..."):
                    try:
                        result  = rag.query(user_input, mode=mode)
                        context = rag.format_context(result)

                        actual_mode = result.get("mode", mode)
                        summaries   = result.get("summaries", [])
                        entities    = result.get("entities", [])
                        hits        = result.get("hits", [])

                        with st.spinner("✍️ Generating answer..."):
                            answer = call_llm(
                                st.session_state.messages,
                                context,
                                actual_mode,
                                st.session_state.provider,
                                st.session_state.api_key
                            )

                        st.markdown(answer)

                        mode_css = f"mode-{actual_mode.split('_')[0]}"
                        st.markdown(f'<span class="{mode_css}">Mode: {actual_mode}</span>', unsafe_allow_html=True)

                        if entities:
                            entity_tags = " ".join(f'<span class="entity-tag">{e}</span>' for e in entities[:15])
                            st.markdown(f"**Entities expanded:** {entity_tags}", unsafe_allow_html=True)

                        if summaries:
                            with st.expander("🌐 Community Summaries Used", expanded=False):
                                for s in summaries:
                                    st.markdown(f'<div class="summary-box">{s}</div>', unsafe_allow_html=True)

                        with st.expander(f"📎 View {len(hits)} Source Excerpts", expanded=False):
                            for j, (text, meta, score) in enumerate(hits, 1):
                                src = meta.get("source", "unknown")
                                ts  = f"{meta.get('start_time', '?')} → {meta.get('end_time', '?')}"
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'<span class="source-badge">📂 {src}</span> '
                                    f'<span class="timestamp-badge">⏱ {ts}</span> '
                                    f'<span style="color:#aaa;font-size:0.75rem"> · Score: {score:.3f}</span>'
                                    f'<br><br>{text}</div>',
                                    unsafe_allow_html=True
                                )

                        st.session_state.messages.append({
                            "role":      "assistant",
                            "content":   answer,
                            "mode":      actual_mode,
                            "summaries": summaries,
                            "entities":  entities,
                            "hits":      hits
                        })

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.exception(e)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # ── Knowledge Graph Tab ────────────────────────────────────────────────────
    with tab_graph:
        st.markdown("### 🕸️ Knowledge Graph Overview")

        gs = rag.get_graph_stats()
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip(
            [c1, c2, c3, c4],
            [("Entities", gs["nodes"]), ("Co-occurrence Edges", gs["edges"]),
             ("Communities", gs["communities"]), ("Graph Density", gs["density"])]
        ):
            col.metric(label, val)

        st.markdown("---")

        col_ent, col_comm = st.columns([1, 2])

        with col_ent:
            st.markdown("#### 🏷️ Top Entities by Mentions")
            top_ents = rag.get_top_entities(n=30)
            if top_ents:
                import pandas as pd
                df_ents = pd.DataFrame(top_ents, columns=["Entity", "Mentions"])
                st.dataframe(df_ents, use_container_width=True, hide_index=True)
            else:
                st.info("No entities extracted yet.")

        with col_comm:
            st.markdown("#### 🔍 Community Clusters")
            communities = rag.get_community_list()
            if communities:
                for comm in communities[:15]:  # show top 15 communities
                    with st.expander(
                        f"Community {comm['id']} · {comm['size']} entities · {comm['num_chunks']} chunks",
                        expanded=False
                    ):
                        entity_tags = " ".join(
                            f'<span class="entity-tag">{e}</span>'
                            for e in comm["top_entities"]
                        )
                        st.markdown(f"**Top entities:** {entity_tags}", unsafe_allow_html=True)
                        st.markdown(f'<div class="summary-box">{comm["summary"]}</div>', unsafe_allow_html=True)
            else:
                st.info("No communities detected yet.")

        if not rag.has_summaries():
            st.warning(
                "Community summaries have not been generated yet. "
                "Use the **Generate Summaries Now** button in the sidebar, or re-ingest with the checkbox enabled."
            )
