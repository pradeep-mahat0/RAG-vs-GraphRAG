"""
Podcast RAG - Streamlit Chat Interface (Multi-Transcript)
Run: streamlit run app.py
"""

import os
import uuid
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎙️ Podcast RAG",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .chunk-box {
        background: #f8f9fa; border-left: 3px solid #667eea;
        padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
        font-size: 0.85rem; color: #444; margin-bottom: 0.5rem;
    }
    .timestamp-badge {
        background: #667eea22; color: #667eea; border-radius: 12px;
        padding: 2px 10px; font-size: 0.78rem; font-weight: 600;
    }
    .source-badge {
        background: #764ba222; color: #764ba2; border-radius: 12px;
        padding: 2px 10px; font-size: 0.78rem; font-weight: 600;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 10px; padding: 1rem; text-align: center;
        border: 1px solid #667eea33;
    }
    .stat-number { font-size: 1.6rem; font-weight: 700; color: #667eea; }
    .stat-label  { font-size: 0.78rem; color: #888; }
    .source-tag  {
        display: inline-block; background: #667eea; color: white;
        border-radius: 8px; padding: 2px 8px; font-size: 0.75rem; margin: 2px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages":          [],
        "rag":               None,
        "ingested":          False,
        "ingest_stats_list": [],
        "active_sources":    [],
        "api_key":           os.getenv("OPENAI_API_KEY", "") or os.getenv("GROQ_API_KEY", ""),
        "provider":          "openai",
        "persist_dir":       f"/tmp/rag_store_{uuid.uuid4().hex}",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(messages: list, context: str, provider: str, api_key: str) -> str:
    active_sources = st.session_state.get("active_sources", [])
    num_sources    = len(active_sources)
    source_list    = ", ".join(active_sources) if active_sources else "unknown"

    if not context or context.strip() == "No relevant excerpts found.":
        return "I couldn't find relevant information in the transcripts to answer this question."

    system_prompt = f"""You are a podcast transcript assistant. Answer questions using the transcript excerpts provided below, drawn from {num_sources} transcript(s): {source_list}.

STRICT RULES:
1. ONLY use information explicitly present in the TRANSCRIPT EXCERPTS below.
2. If the excerpts do not contain a clear answer, say: "This topic is not covered in the available transcripts."
3. When answering, cite which source and approximate timestamp (e.g., "In [source_name] around 12:34, ...").
4. If multiple transcripts discuss the same topic, compare or synthesize their perspectives.
5. Do NOT guess, infer, or use outside knowledge under any circumstances.
6. Keep answers concise and grounded.

TRANSCRIPT EXCERPTS:
{context}"""

    full_messages = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
    full_messages.append({"role": "user", "content": messages[-1]["content"]})

    if provider == "groq":
        from groq import Groq
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}] + full_messages,
            max_tokens=1024, temperature=0.3,
        )
        return response.choices[0].message.content

    else:  # openai
        from openai import OpenAI
        client   = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + full_messages,
            max_tokens=1024, temperature=0.3,
        )
        return response.choices[0].message.content


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.session_state.provider = st.selectbox(
        "LLM Provider", ["openai", "groq"],
        index=0 if st.session_state.provider == "openai" else 1,
        help="OpenAI gpt-4o-mini or Groq llama-3.3-70b (free & fast)."
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

    # ── Multi-file Upload ──────────────────────────────────────────────────────
    st.markdown("## 📂 Upload Transcripts")
    uploaded_files = st.file_uploader(
        "Upload .srt / .txt files",
        type=["srt", "txt"],
        accept_multiple_files=True,
        help="Upload multiple transcripts and ask questions across all of them."
    )

    if uploaded_files:
        already  = set(st.session_state.active_sources)
        pending  = [f for f in uploaded_files if Path(f.name).stem not in already]

        btn_label = (
            f"🚀 Process {len(pending)} new transcript{'s' if len(pending) != 1 else ''}"
            if pending else "✅ All uploaded files already loaded"
        )

        if st.button(btn_label, use_container_width=True, type="primary", disabled=not pending):
            # Create RAG object once and reuse across uploads
            if st.session_state.rag is None:
                from rag_pipeline import PodcastRAG
                st.session_state.rag = PodcastRAG(
                    persist_dir=st.session_state.persist_dir,
                    chunk_size=chunk_size, overlap=overlap,
                    top_k_retrieve=top_k, top_n_rerank=top_n
                )

            progress = st.progress(0.0, text="Starting…")
            new_stats = []
            for i, uf in enumerate(pending):
                progress.progress((i + 0.5) / len(pending), text=f"Processing: {uf.name}")
                # Use UUID to avoid temp file conflicts between concurrent sessions
                tmp_path = Path(f"/tmp/{uuid.uuid4().hex}_{uf.name}")
                try:
                    tmp_path.write_bytes(uf.getvalue())
                    stats = st.session_state.rag.ingest(str(tmp_path))
                    new_stats.append(stats)
                except Exception as e:
                    st.error(f"❌ Error processing {uf.name}: {e}")
                finally:
                    tmp_path.unlink(missing_ok=True)  # clean up tmp file

            progress.progress(1.0, text="Done!")
            st.session_state.ingested          = True
            st.session_state.ingest_stats_list.extend(new_stats)
            st.session_state.active_sources    = list(st.session_state.rag._ingested_sources.keys())
            st.session_state.restored          = False
            st.success(f"✅ {len(new_stats)} transcript(s) ready!")
            st.rerun()

    # ── Active Sources Panel ───────────────────────────────────────────────────
    if st.session_state.active_sources:
        st.divider()
        st.markdown("## 📚 Active Transcripts")
        for src in st.session_state.active_sources:
            st.markdown(f'<span class="source-tag">🎙 {src}</span>', unsafe_allow_html=True)

    # ── Index Stats ────────────────────────────────────────────────────────────
    if st.session_state.ingest_stats_list:
        st.divider()
        st.markdown("## 📊 Index Stats")
        total_chunks = sum(s.get("chunks", 0) for s in st.session_state.ingest_stats_list)
        col1, col2 = st.columns(2)
        col1.markdown(f"""<div class="stat-box">
            <div class="stat-number">{len(st.session_state.active_sources)}</div>
            <div class="stat-label">Sources</div>
        </div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="stat-box">
            <div class="stat-number">{total_chunks}</div>
            <div class="stat-label">Chunks</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2)
    if col_a.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if col_b.button("♻️ Reset All", use_container_width=True):
        st.session_state.messages          = []
        st.session_state.rag               = None
        st.session_state.ingested          = False
        st.session_state.ingest_stats_list = []
        st.session_state.active_sources    = []
        st.session_state.persist_dir       = f"/tmp/rag_store_{uuid.uuid4().hex}"
        st.rerun()


# ─── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🎙️ Podcast RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Multi-transcript chat · Dense + BM25 Hybrid Retrieval · Cross-Encoder Reranking</div>',
    unsafe_allow_html=True
)

if not st.session_state.ingested:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📥 Step 1: Upload**
        Upload one or more `.srt` / `.txt` transcripts in the sidebar.
        """)
    with col2:
        st.markdown("""
        **⚙️ Step 2: Process**
        Each transcript is cleaned, chunked, embedded, and added to the shared index.
        """)
    with col3:
        st.markdown("""
        **💬 Step 3: Chat**
        Ask questions across all transcripts simultaneously.
        """)
    st.markdown("---")
    st.info("👈 Upload transcript files in the sidebar to get started.")

else:
    sources = st.session_state.active_sources
    src_str = " · ".join(f"`{s}`" for s in sources)
    st.markdown(f"**📄 Active ({len(sources)} transcript{'s' if len(sources) != 1 else ''}):** {src_str}")
    st.markdown("---")

    # ── Message History ────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} Source Excerpts", expanded=False):
                    for text, meta, score in msg["sources"]:
                        src = meta.get("source", "?")
                        ts  = f"{meta.get('start_time', '?')} → {meta.get('end_time', '?')}"
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<span class="source-badge">📂 {src}</span> '
                            f'<span class="timestamp-badge">⏱ {ts}</span>'
                            f'<span style="color:#aaa;font-size:0.75rem"> · {score:.3f}</span>'
                            f'<br><br>{text}</div>',
                            unsafe_allow_html=True
                        )

    # ── Suggestion Buttons ────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What are the main themes discussed across all episodes?",
            "How do the guests' views on AI compare to each other?",
            "What did the speakers say about self-driving / autonomous vehicles?",
            "What are the key insights about the future of AI?",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["_prefill"] = sug
                st.rerun()

    # ── Chat Input (reads suggestion prefill if set) ───────────────────────────
    user_input = st.chat_input("Ask anything across all transcripts...")
    if not user_input and "_prefill" in st.session_state:
        user_input = st.session_state.pop("_prefill")

    if user_input:
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your API key in the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Retrieving relevant excerpts…"):
                    hits    = st.session_state.rag.retrieve(user_input)
                    context = st.session_state.rag.format_context(hits)

                with st.spinner("✍️ Generating answer…"):
                    answer = call_llm(
                        st.session_state.messages, context,
                        st.session_state.provider, st.session_state.api_key
                    )

                st.markdown(answer)

                if hits:
                    with st.expander(f"📎 {len(hits)} Source Excerpts", expanded=False):
                        for text, meta, score in hits:
                            src = meta.get("source", "?")
                            ts  = f"{meta.get('start_time', '?')} → {meta.get('end_time', '?')}"
                            st.markdown(
                                f'<div class="chunk-box">'
                                f'<span class="source-badge">📂 {src}</span> '
                                f'<span class="timestamp-badge">⏱ {ts}</span>'
                                f'<span style="color:#aaa;font-size:0.75rem"> · {score:.3f}</span>'
                                f'<br><br>{text}</div>',
                                unsafe_allow_html=True
                            )

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": hits
                })

            except Exception as e:
                error_msg = f"❌ {type(e).__name__}: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
