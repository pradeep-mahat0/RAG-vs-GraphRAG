"""
GraphRAG Evaluation Dashboard
==============================
Streamlit app to evaluate and visualize GraphRAG performance across
Local / Global / Hybrid modes using RAGAS metrics.

Run:  streamlit run eval_app.py
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT         = Path(__file__).parent
EVAL_STORE   = ROOT / "eval_store"
RESULTS_PATH = EVAL_STORE / "results.json"

sys.path.insert(0, str(ROOT / "GraphRAG"))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .eval-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 60%, #4facfe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .section-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .mode-local   { background:#11998e22; color:#11998e; border-radius:6px; padding:2px 10px; font-weight:600; font-size:0.85rem; }
    .mode-global  { background:#6366f122; color:#6366f1; border-radius:6px; padding:2px 10px; font-weight:600; font-size:0.85rem; }
    .mode-hybrid  { background:#f5922122; color:#f59221; border-radius:6px; padding:2px 10px; font-weight:600; font-size:0.85rem; }
    .q-factual    { background:#dbeafe; color:#1d4ed8; border-radius:6px; padding:2px 8px; font-size:0.78rem; }
    .q-relational { background:#fef3c7; color:#92400e; border-radius:6px; padding:2px 8px; font-size:0.78rem; }
    .q-thematic   { background:#f3e8ff; color:#6b21a8; border-radius:6px; padding:2px 8px; font-size:0.78rem; }
    .answer-box   { padding:0.75rem 1rem; border-radius:0 8px 8px 0; font-size:0.88rem; line-height:1.55; }
    .winner-chip  { background:#22c55e; color:white; border-radius:6px; padding:1px 8px; font-size:0.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

MODE_COLORS = {
    "local":  "#11998e",
    "global": "#6366f1",
    "hybrid": "#f59221",
}
MODE_LABELS = {"local": "Local", "global": "Global", "hybrid": "Hybrid"}
METRICS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}
METRIC_EXPLAINERS = {
    "Faithfulness":      "Is the answer grounded in retrieved context? Penalizes hallucination.",
    "Answer Relevancy":  "Does the answer directly address the question asked?",
    "Context Precision": "Of the chunks retrieved, how many were actually relevant?",
    "Context Recall":    "Did retrieval surface ALL the information needed to answer?",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    provider = st.selectbox("LLM Provider", ["gemini", "groq", "openai"])
    env_key  = os.getenv(
        "GEMINI_API_KEY" if provider == "gemini" else (
            "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
        ),
        ""
    )
    api_key  = st.text_input(
        "API Key", value=env_key, type="password",
        placeholder="Loaded from .env" if env_key else "Enter API key...",
    )

    st.divider()
    force_rerun = st.checkbox("Force re-run (clears cache)", value=False)
    run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("#### RAGAS Metrics")
    for label, desc in METRIC_EXPLAINERS.items():
        with st.expander(label):
            st.caption(desc)

    st.divider()
    st.markdown("#### Question Types")
    st.markdown("""
- 🔵 **Factual** (3) — single-source specific facts
- 🟡 **Relational** (4) — entity connections across docs
- 🟣 **Thematic** (3) — synthesis across all 3 transcripts
""")


# ── Run evaluation ────────────────────────────────────────────────────────────

if run_btn:
    final_key = api_key or env_key
    if not final_key:
        st.sidebar.error("API key required.")
    else:
        if force_rerun:
            for f in ["results.json", "ground_truths.json"]:
                p = EVAL_STORE / f
                if p.exists(): p.unlink()

        from evaluate import run_evaluation

        st.markdown("### Running Evaluation...")
        progress   = st.progress(0, text="Starting...")
        log_holder = st.empty()
        log_lines  = []

        def on_progress(msg):
            log_lines.append(msg)
            pct = min(len(log_lines) / 55, 0.95)
            progress.progress(pct, text=msg[:90])
            log_holder.caption("\n".join(log_lines[-4:]))

        try:
            run_evaluation(provider, final_key, progress_fn=on_progress)
            progress.progress(1.0, text="Complete!")
            log_holder.empty()
            st.success("Evaluation complete! Results saved.")
            st.rerun()
        except Exception as e:
            progress.empty()
            st.error(f"Evaluation failed: {e}")
            st.exception(e)
            st.stop()


# ── Load results ──────────────────────────────────────────────────────────────

def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return None

results = load_results()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="eval-header">GraphRAG Evaluation Dashboard</div>', unsafe_allow_html=True)
st.caption("Evaluating Local / Global / Hybrid modes across 10 questions using RAGAS — powered by 3 Lex Fridman podcast transcripts")

if results is None:
    st.info("No evaluation results yet. Click **▶ Run Evaluation** in the sidebar to get started.")
    st.markdown("""
**What happens when you run:**
1. GraphRAG ingests all 3 transcripts and builds the entity knowledge graph *(uses disk cache on re-runs)*
2. Reference answers are generated using Hybrid mode as oracle *(cached after first run)*
3. All 3 modes answer 10 test questions — 3 factual, 4 relational, 3 thematic
4. RAGAS scores each mode on Faithfulness, Answer Relevancy, Context Precision, Context Recall

**Estimated time:** ~8 min first run · ~3 min on re-runs (cached embeddings)
""")
    st.stop()


# ── Parse data ────────────────────────────────────────────────────────────────

meta      = results["metadata"]
questions = results["questions"]
res       = results["results"]   # {mode: {scores, samples}}
modes     = list(MODE_LABELS.keys())
metric_labels = list(METRICS.values())

# Build aggregate scores dataframe
score_rows = []
for mode in modes:
    s = res.get(mode, {}).get("scores", {})
    row = {"Mode": MODE_LABELS[mode]}
    for key, label in METRICS.items():
        row[label] = s.get(key)
    score_rows.append(row)
scores_df = pd.DataFrame(score_rows).set_index("Mode")


# ── Section 1: Run metadata ───────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transcripts Loaded", len(meta["transcripts"]))
c2.metric("Test Questions",     meta["n_questions"])
c3.metric("Modes Evaluated",    3)
c4.metric("Evaluated",          meta["timestamp"].split(" ")[0])

st.divider()


# ── Section 2: Overall metric comparison ─────────────────────────────────────

st.markdown('<div class="section-title">Overall RAGAS Scores — All Modes</div>', unsafe_allow_html=True)
st.caption("Average across all 10 questions. Higher = better (0–1 scale).")

fig_bar = go.Figure()
for mode in modes:
    s = res.get(mode, {}).get("scores", {})
    fig_bar.add_trace(go.Bar(
        name=MODE_LABELS[mode],
        x=metric_labels,
        y=[s.get(k) for k in METRICS],
        marker_color=MODE_COLORS[mode],
        text=[f"{s.get(k):.3f}" if s.get(k) is not None else "N/A" for k in METRICS],
        textposition="outside",
        width=0.22,
    ))

fig_bar.update_layout(
    barmode="group", height=400,
    yaxis=dict(range=[0, 1.15], title="Score (0–1)", gridcolor="#f0f0f0"),
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=30, l=40, r=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_bar, use_container_width=True)

# Score table with colour highlights
st.dataframe(
    scores_df.style
        .format("{:.3f}", na_rep="N/A")
        .highlight_max(axis=0, color="#bbf7d0")
        .highlight_min(axis=0, color="#fecaca"),
    use_container_width=True,
)
st.caption("🟢 Best per metric &nbsp; 🔴 Worst per metric")

st.divider()


# ── Section 3: Radar — mode strength profiles ─────────────────────────────────

st.markdown('<div class="section-title">Mode Capability Profiles</div>', unsafe_allow_html=True)
st.caption("Each axis is a RAGAS metric. Larger area = stronger overall performance.")

fig_radar = go.Figure()
for mode in modes:
    s    = res.get(mode, {}).get("scores", {})
    vals = [s.get(k) or 0 for k in METRICS]
    vals_closed = vals + [vals[0]]
    cats_closed = metric_labels + [metric_labels[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill="toself", name=MODE_LABELS[mode],
        line_color=MODE_COLORS[mode],
        fillcolor=MODE_COLORS[mode],
        opacity=0.25,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    margin=dict(t=20, b=60),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()


# ── Section 4: Context Recall by question type ────────────────────────────────

st.markdown('<div class="section-title">Context Recall by Question Type</div>', unsafe_allow_html=True)
st.caption("Shows which mode retrieves the most complete information for each question category.")

q_by_type = {}
for q in questions:
    q_by_type.setdefault(q["type"], []).append(q["question"])

type_rows = []
for q_type in ["factual", "relational", "thematic"]:
    type_qs = q_by_type.get(q_type, [])
    for mode in modes:
        per_q = res.get(mode, {}).get("scores", {}).get("per_question", [])
        vals  = [r.get("context_recall") for r in per_q
                 if r.get("user_input") in type_qs and r.get("context_recall") is not None]
        avg   = round(sum(vals) / len(vals), 3) if vals else None
        type_rows.append({"Question Type": q_type.capitalize(), "Mode": MODE_LABELS[mode], "Context Recall": avg})

type_df = pd.DataFrame(type_rows)
if type_df["Context Recall"].notna().any():
    fig_type = px.bar(
        type_df, x="Question Type", y="Context Recall", color="Mode",
        barmode="group",
        color_discrete_map={MODE_LABELS[m]: MODE_COLORS[m] for m in modes},
        height=360,
        text="Context Recall",
    )
    fig_type.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_type.update_layout(
        yaxis=dict(range=[0, 1.15], gridcolor="#f0f0f0"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_type, use_container_width=True)
    st.caption("Expected: Local wins on Factual/Relational · Global wins on Thematic · Hybrid balances both")
else:
    st.info("Per-question data not available.")

st.divider()


# ── Section 5: Graph-Specific Structural Metrics ─────────────────────────────

st.markdown('<div class="section-title">Graph-Specific Structural Metrics</div>', unsafe_allow_html=True)
st.caption("Metrics RAGAS cannot measure — they inspect the graph structure directly.")

# Helper: extract graph_metrics per sample
def get_graph_metric(mode, metric_key):
    samples = res.get(mode, {}).get("samples", [])
    vals = [s.get("graph_metrics", {}).get(metric_key) for s in samples
            if s.get("graph_metrics", {}).get(metric_key) is not None]
    return vals

def avg_by_type(mode, metric_key):
    samples = res.get(mode, {}).get("samples", [])
    by_type = {}
    for s in samples:
        t   = s.get("type", "unknown")
        val = s.get("graph_metrics", {}).get(metric_key)
        if val is not None:
            by_type.setdefault(t, []).append(val)
    return {t: round(sum(v) / len(v), 3) for t, v in by_type.items()}

# ── Metric 1: Entity Coverage Rate ───────────────────────────────────────────

st.markdown("#### Entity Coverage Rate")
st.caption("% of ground-truth entities that appear in the retrieved context. Higher = retrieval is surfacing the right actors and concepts.")

ecr_rows = []
for mode in modes:
    by_type = avg_by_type(mode, "entity_coverage_rate")
    for q_type in ["factual", "relational", "thematic"]:
        ecr_rows.append({
            "Mode": MODE_LABELS[mode],
            "Question Type": q_type.capitalize(),
            "Entity Coverage Rate": by_type.get(q_type),
        })

ecr_df = pd.DataFrame(ecr_rows)
if ecr_df["Entity Coverage Rate"].notna().any():
    fig_ecr = px.bar(
        ecr_df, x="Question Type", y="Entity Coverage Rate", color="Mode",
        barmode="group",
        color_discrete_map={MODE_LABELS[m]: MODE_COLORS[m] for m in modes},
        height=340, text="Entity Coverage Rate",
    )
    fig_ecr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_ecr.update_layout(
        yaxis=dict(range=[0, 1.2], gridcolor="#f0f0f0"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ecr, use_container_width=True)
else:
    st.info("Entity coverage data not available. Re-run evaluation to compute.")

# ── Metric 2: Graph Utilization ───────────────────────────────────────────────

st.markdown("#### Graph Utilization (Local & Hybrid only)")
st.caption("Fraction of the final retrieved chunks that came exclusively from 1-hop entity graph expansion, not from base dense+BM25 retrieval.")

gu_col1, gu_col2, gu_col3 = st.columns(3)
for col, mode in zip([gu_col1, gu_col2, gu_col3], modes):
    vals = get_graph_metric(mode, "graph_utilization")
    with col:
        if vals and mode in ("local", "hybrid"):
            avg_gu = round(sum(vals) / len(vals), 3)
            st.metric(
                label=f"{MODE_LABELS[mode]} — Graph Utilization",
                value=f"{avg_gu:.1%}",
                help="Avg across all questions. How much of the final context was contributed by graph expansion?"
            )
            exp_vals = get_graph_metric(mode, "graph_expanded_chunks")
            base_vals = get_graph_metric(mode, "base_retrieved_chunks")
            if exp_vals and base_vals:
                st.caption(
                    f"Avg {sum(exp_vals)/len(exp_vals):.1f} graph-expanded chunks · "
                    f"{sum(base_vals)/len(base_vals):.1f} base chunks per query"
                )
        elif mode == "global":
            st.metric(label="Global — Graph Utilization", value="N/A",
                      help="Global mode uses community summaries, not entity graph expansion.")

# ── Metric 3: Community Coherence ─────────────────────────────────────────────

st.markdown("#### Community Coherence (Global & Hybrid only)")
st.caption(
    "Avg % of a community's top entities explicitly mentioned in its own LLM-generated summary. "
    "High coherence = the LLM summary accurately captured the key entities in that cluster."
)

cc_col1, cc_col2, cc_col3 = st.columns(3)
for col, mode in zip([cc_col1, cc_col2, cc_col3], modes):
    vals = get_graph_metric(mode, "community_coherence")
    with col:
        if vals and mode in ("global", "hybrid"):
            avg_cc = round(sum(vals) / len(vals), 3)
            st.metric(
                label=f"{MODE_LABELS[mode]} — Community Coherence",
                value=f"{avg_cc:.1%}",
                help="Avg across queries where community summaries were used."
            )
            cu_vals = get_graph_metric(mode, "communities_used")
            if cu_vals:
                st.caption(f"Avg {sum(cu_vals)/len(cu_vals):.1f} communities used per query")
        elif mode == "local":
            st.metric(label="Local — Community Coherence", value="N/A",
                      help="Local mode does not use community summaries.")

st.divider()


# ── Section 6: Question-by-question answers ───────────────────────────────────

st.markdown('<div class="section-title">Question-by-Question Answers</div>', unsafe_allow_html=True)
st.caption("Expand any question to see how each mode answered it and what the ground truth reference was.")

type_icon  = {"factual": "🔵", "relational": "🟡", "thematic": "🟣"}
type_class = {"factual": "q-factual", "relational": "q-relational", "thematic": "q-thematic"}

for q in questions:
    qid    = q["id"]
    qtype  = q["type"]
    icon   = type_icon.get(qtype, "")
    badge  = f'<span class="{type_class[qtype]}">{qtype.upper()}</span>'
    label  = f"{icon} {qid} — {q['question'][:75]}..."

    with st.expander(label):
        st.markdown(f"**Full question:** {q['question']}")
        st.markdown(f"{badge} &nbsp; Expected best mode: `{q.get('expected_best_mode', '—')}`",
                    unsafe_allow_html=True)
        st.divider()

        # Ground truth
        gt_text = res.get("hybrid", {}).get("samples", [])
        gt = next((s["ground_truth"] for s in gt_text if s["id"] == qid), "")
        if gt:
            st.markdown("**Reference Answer (Hybrid Oracle)**")
            st.markdown(
                f'<div class="answer-box" style="background:#f0fdf4; border-left:3px solid #22c55e;">{gt}</div>',
                unsafe_allow_html=True
            )
            st.markdown("")

        # Per-mode answers
        cols = st.columns(3)
        for col, mode in zip(cols, modes):
            samples = res.get(mode, {}).get("samples", [])
            sample  = next((s for s in samples if s["id"] == qid), None)
            color   = MODE_COLORS[mode]

            with col:
                st.markdown(f'<span class="mode-{mode}">{MODE_LABELS[mode]}</span>',
                            unsafe_allow_html=True)
                answer = sample["answer"] if sample else "No answer."
                st.markdown(
                    f'<div class="answer-box" style="background:{color}0d; border-left:3px solid {color};">'
                    f'{answer}</div>',
                    unsafe_allow_html=True
                )
                if sample:
                    st.caption(f"{len(sample.get('contexts', []))} context(s) retrieved")

st.divider()


# ── Section 7: Key insights ───────────────────────────────────────────────────

st.markdown('<div class="section-title">How to Read These Results</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Context Recall — the most revealing metric**

This asks: *did the system retrieve everything it needed to fully answer the question?*

- **Factual questions** → all 3 modes retrieve the relevant chunk. Similar scores.
- **Relational questions** → Local's entity graph expansion pulls in connected chunks
  that a flat search would miss. Local recall should be higher.
- **Thematic questions** → Global's community summaries cover the full topic landscape
  across all 3 transcripts. Global recall should peak here.
""")

with col2:
    st.markdown("""
**Context Precision — the trade-off**

Local mode adds extra chunks via 1-hop entity expansion. Some of those
neighboring chunks are less directly relevant to the query — this can
lower Precision slightly even while Recall goes up. This is the
classic retrieval precision/recall trade-off made visible.

**Faithfulness — should be high across all modes**

Both modes use cross-encoder reranking and strict "answer only from
context" prompts. Low faithfulness = the LLM is drifting beyond the
retrieved chunks.

**Hybrid — best of both worlds**

Hybrid combines Local's precision on specific questions with Global's
breadth on thematic ones. It should score consistently across all types.
""")

st.divider()
st.caption(f"Provider: {meta['provider']} · {meta['n_questions']} questions · {meta['timestamp']}")
