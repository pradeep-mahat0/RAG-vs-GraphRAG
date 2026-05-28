"""
GraphRAG Evaluation — Core Engine
===================================
Evaluates GraphRAG across 3 query modes using RAGAS metrics.

Modes:
  Local  → entity-graph-expanded retrieval  (factual / relational questions)
  Global → community-summary-based retrieval (thematic / broad questions)
  Hybrid → combines both                     (best overall)

RAGAS Metrics:
  Faithfulness      — does the answer stick to the retrieved context?
  Answer Relevancy  — does the answer actually address the question?
  Context Precision — of chunks retrieved, how many were truly relevant?
  Context Recall    — did retrieval surface ALL info needed to answer?

Evaluation Flow:
  1. Load GraphRAG pipeline (ingests 3 transcripts, uses disk cache on re-runs)
  2. Generate reference answers using Hybrid as oracle → saved to eval_store/ground_truths.json
  3. Run Local / Global / Hybrid on each of 10 test questions
  4. Score with RAGAS → save to eval_store/results.json

Run:
  python evaluate.py --provider openai --api_key sk-...
  python evaluate.py --provider groq   --api_key gsk-...
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Callable

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "GraphRAG"))

from dotenv import load_dotenv
load_dotenv()

# ── Transcript paths ──────────────────────────────────────────────────────────

TRANSCRIPT_PATHS = [
    str(ROOT / "Jensen Huang_ NVIDIA - The 4 Trillion Company & the AI Revolution _ Lex Fridman Podcast -494.srt"),
    str(ROOT / "Sam Altman_ OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power & AGI _ Lex Fridman Podcast -419.srt"),
    str(ROOT / "Elon Musk_ War, AI, Aliens, Politics, Physics, Video Games, and Humanity _ Lex Fridman Podcast -400.srt"),
]

EVAL_STORE = ROOT / "eval_store"

# ── Test Questions ─────────────────────────────────────────────────────────────
# 10 questions across 3 types — each type designed to stress a different mode.
#
#   factual    → specific single-source fact   → Local should handle well
#   relational → entity connection across docs → Local graph expansion wins
#   thematic   → synthesis across all 3 docs   → Global community summaries win

TEST_QUESTIONS = [
    # Factual (3)
    {
        "id": "Q01",
        "question": "How does Jensen Huang describe NVIDIA's shift from chip-scale to rack-scale design and why does he say it became necessary?",
        "type": "factual",
        "expected_best_mode": "local",
    },
    {
        "id": "Q02",
        "question": "What does Sam Altman say about the OpenAI board saga and how it affected him personally and professionally?",
        "type": "factual",
        "expected_best_mode": "local",
    },
    {
        "id": "Q03",
        "question": "What does Elon Musk say about the role of AI in modern warfare and its implications for geopolitics?",
        "type": "factual",
        "expected_best_mode": "local",
    },
    # Relational (4)
    {
        "id": "Q04",
        "question": "How are NVIDIA's GPU architecture, large-scale distributed training, and the AI infrastructure buildout connected across the podcast discussions?",
        "type": "relational",
        "expected_best_mode": "local",
    },
    {
        "id": "Q05",
        "question": "How are Sam Altman, Elon Musk, and Ilya Sutskever connected and how did their relationships shape OpenAI's direction?",
        "type": "relational",
        "expected_best_mode": "local",
    },
    {
        "id": "Q06",
        "question": "How do Jensen Huang and Sam Altman compare in their views on what compute scaling means for reaching AGI?",
        "type": "relational",
        "expected_best_mode": "local",
    },
    {
        "id": "Q07",
        "question": "What connections exist between robotics, physical AI, and the ambitions of NVIDIA, OpenAI, and Elon Musk's companies across the three conversations?",
        "type": "relational",
        "expected_best_mode": "hybrid",
    },
    # Thematic (3)
    {
        "id": "Q08",
        "question": "What are the main recurring themes about the risks and benefits of AGI across all three podcast conversations?",
        "type": "thematic",
        "expected_best_mode": "global",
    },
    {
        "id": "Q09",
        "question": "How do all three guests collectively discuss the concentration of AI power and who should control it?",
        "type": "thematic",
        "expected_best_mode": "global",
    },
    {
        "id": "Q10",
        "question": "What common perspective on the relationship between compute, data, and the pace of AI progress emerges across the three interviews?",
        "type": "thematic",
        "expected_best_mode": "global",
    },
]


# ── LLM helpers ───────────────────────────────────────────────────────────────

def call_llm(prompt: str, provider: str, api_key: str, max_tokens: int = 400) -> str:
    if provider == "groq":
        from groq import Groq
        resp = Groq(api_key=api_key).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    elif provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,
            )
        )
        return resp.text.strip()
    else:
        from openai import OpenAI
        resp = OpenAI(api_key=api_key).chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.1,
        )
        return resp.choices[0].message.content.strip()


def generate_answer(question: str, context: str, mode: str,
                    provider: str, api_key: str) -> str:
    mode_note = {
        "local":  "You have entity-graph-expanded excerpts relevant to the question.",
        "global": "You have thematic community summaries and excerpts. Synthesize the themes.",
        "hybrid": "You have both entity-specific excerpts and thematic community summaries.",
    }.get(mode, "")

    prompt = f"""You are analyzing podcast transcripts. {mode_note}

RULES:
1. Answer ONLY using the CONTEXT below.
2. If context is insufficient say: "This topic is not sufficiently covered in the transcripts."
3. Cite source and approximate timestamp where possible.
4. Be thorough — use all relevant information.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""
    return call_llm(prompt, provider, api_key, max_tokens=600)


# ── Pipeline setup ────────────────────────────────────────────────────────────

def setup_graphrag(llm_fn: Callable, progress_fn: Optional[Callable] = None):
    from graph_pipeline import GraphRAG
    import glob

    grag = GraphRAG(persist_dir=str(EVAL_STORE / "graph_rag_store"))

    if grag.load_state():
        if progress_fn: progress_fn("GraphRAG: restored from cache")
        if grag.has_summaries():
            grag._cache_summary_embeddings()
        return grag

    # Match transcripts dynamically in the ROOT directory
    all_files = glob.glob(str(ROOT / "*.srt")) + glob.glob(str(ROOT / "*.txt"))
    resolved_paths = []
    
    # Define keywords to match each transcript
    keywords_map = {
        "Jensen Huang": ["jensen", "huang", "nvidia", "494"],
        "Sam Altman": ["altman", "openai", "gpt-5", "419", "future"],
        "Elon Musk": ["elon", "musk", "war", "aliens", "400"]
    }
    
    for speaker, keywords in keywords_map.items():
        matched = False
        # Try to find a matching file in the root
        for f in all_files:
            fname = os.path.basename(f).lower()
            if any(kw in fname for kw in keywords):
                resolved_paths.append(f)
                matched = True
                if progress_fn: progress_fn(f"Matched transcript for {speaker}: {os.path.basename(f)}")
                break
        
        # If no match in root, check if the default filename in TRANSCRIPT_PATHS exists
        if not matched:
            for default_path in TRANSCRIPT_PATHS:
                default_fname = os.path.basename(default_path).lower()
                if any(kw in default_fname for kw in keywords) and Path(default_path).exists():
                    resolved_paths.append(default_path)
                    matched = True
                    if progress_fn: progress_fn(f"Found default transcript for {speaker}")
                    break
            
        if not matched:
            if progress_fn: progress_fn(f"⚠️ Warning: Transcript for {speaker} not found in root directory. Skipping from evaluation dataset.")

    if not resolved_paths:
        raise FileNotFoundError(
            f"No transcripts found in {ROOT}. Please place at least one podcast transcript (.srt or .txt) in the directory."
        )

    for path in resolved_paths:
        if progress_fn: progress_fn(f"Ingesting: {Path(path).name[:50]}...")
        grag.ingest(path, llm_fn=llm_fn)

    return grag


# ── Graph-Specific Structural Metrics ────────────────────────────────────────

def compute_graph_metrics(grag, result: dict, question: str, mode: str,
                          ground_truth: str = "") -> dict:
    """
    Three graph-specific metrics that RAGAS cannot measure:

    1. entity_coverage_rate  — % of ground-truth entities present in retrieved context.
                               Tells you whether retrieval is surfacing the right actors/concepts.

    2. graph_utilization     — fraction of final answer chunks that came from graph expansion
                               (i.e. were NOT in the base dense+BM25 retrieval).
                               Only meaningful for local / hybrid modes.

    3. community_coherence   — avg % of a community's top entities explicitly mentioned in
                               its own LLM-generated summary.
                               Only meaningful for global / hybrid modes.
    """
    metrics = {}

    # ── 1. Entity Coverage Rate (all modes) ──────────────────────────────────
    if ground_truth:
        gt_entities  = set(grag.extract_entities(ground_truth))
        ctx_entities = set()
        for text, _, _ in result["hits"]:
            ctx_entities.update(grag.extract_entities(text))
        for summary in result.get("summaries", []):
            ctx_entities.update(grag.extract_entities(summary))

        if gt_entities:
            covered = gt_entities & ctx_entities
            metrics["entity_coverage_rate"]  = round(len(covered) / len(gt_entities), 3)
            metrics["gt_entities_total"]     = len(gt_entities)
            metrics["gt_entities_covered"]   = len(covered)
        else:
            metrics["entity_coverage_rate"] = None

    # ── 2. Graph Utilization (local / hybrid only) ────────────────────────────
    if mode in ("local", "hybrid"):
        try:
            from graph_pipeline import reciprocal_rank_fusion
            dense  = grag.vector_store.dense_search(question, top_k=grag.top_k_retrieve)
            sparse = grag.bm25_index.search(question, top_k=grag.top_k_retrieve)
            fused  = reciprocal_rank_fusion(dense, sparse)
            # All chunk keys reachable by base retrieval (no graph expansion)
            base_keys = {
                f"{meta.get('source', '')}_{meta.get('chunk_id', '')}"
                for _, meta, _ in fused
            }
            final_hits = result["hits"]
            if final_hits:
                graph_only = [
                    1 for _, meta, _ in final_hits
                    if f"{meta.get('source', '')}_{meta.get('chunk_id', '')}" not in base_keys
                ]
                metrics["graph_utilization"]     = round(len(graph_only) / len(final_hits), 3)
                metrics["graph_expanded_chunks"] = len(graph_only)
                metrics["base_retrieved_chunks"] = len(final_hits) - len(graph_only)
            else:
                metrics["graph_utilization"] = 0.0
        except Exception as e:
            metrics["graph_utilization"]       = None
            metrics["graph_utilization_error"] = str(e)

    # ── 3. Community Coherence (global / hybrid only) ────────────────────────
    if mode in ("global", "hybrid"):
        summaries_used = result.get("summaries", [])
        metrics["communities_used"] = len(summaries_used)
        if summaries_used:
            coherence_scores = []
            for summary_text in summaries_used:
                for cd in grag._communities.values():
                    if cd.get("summary") == summary_text:
                        top_ents = cd["entities"][:10]
                        if top_ents:
                            summary_lower = summary_text.lower()
                            mentioned = sum(1 for e in top_ents if e.lower() in summary_lower)
                            coherence_scores.append(mentioned / len(top_ents))
                        break
            if coherence_scores:
                metrics["community_coherence"]   = round(sum(coherence_scores) / len(coherence_scores), 3)
                metrics["communities_evaluated"] = len(coherence_scores)

    return metrics


# ── Query each mode ───────────────────────────────────────────────────────────

def query_mode(grag, question: str, mode: str, provider: str, api_key: str,
               ground_truth: str = "") -> dict:
    result    = grag.query(question, mode=mode)
    chunks    = [text for text, _, _ in result["hits"]]
    summaries = result.get("summaries", [])
    # Summaries first so RAGAS context recall picks up thematic info
    contexts  = (summaries + chunks) if summaries else chunks
    answer    = generate_answer(question, grag.format_context(result), mode, provider, api_key)
    graph_metrics = compute_graph_metrics(grag, result, question, mode, ground_truth)
    return {"answer": answer, "contexts": contexts, "graph_metrics": graph_metrics}


# ── Ground truth generation ───────────────────────────────────────────────────

def get_ground_truths(grag, provider: str, api_key: str,
                      progress_fn: Optional[Callable] = None) -> dict:
    """Use Hybrid as oracle to generate reference answers. Cached after first run."""
    path = EVAL_STORE / "ground_truths.json"
    if path.exists():
        if progress_fn: progress_fn("Ground truths: loaded from cache")
        return json.loads(path.read_text())

    if progress_fn: progress_fn("Generating ground truths via Hybrid oracle...")
    gt = {}
    for i, q in enumerate(TEST_QUESTIONS):
        if progress_fn: progress_fn(f"Ground truth {i+1}/{len(TEST_QUESTIONS)}: {q['id']}")
        result  = grag.query(q["question"], mode="hybrid")
        context = grag.format_context(result)
        prompt  = f"""Write a comprehensive REFERENCE ANSWER for the question below.
Use ALL relevant info from the context. Be specific (2-4 sentences min). Cite sources/timestamps.

QUESTION: {q["question"]}
CONTEXT:\n{context}
REFERENCE ANSWER:"""
        gt[q["id"]] = call_llm(prompt, provider, api_key, max_tokens=500)
        time.sleep(0.5)

    path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    return gt


# ── RAGAS scoring ─────────────────────────────────────────────────────────────

def score_with_ragas(samples: list, provider: str, api_key: str) -> dict:
    try:
        import warnings, pandas as pd
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        raise ImportError("Run: pip install ragas langchain-openai")

    if provider == "groq":
        from langchain_groq import ChatGroq
        ragas_llm = LangchainLLMWrapper(
            ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
        )
    elif provider == "gemini":
        from langchain_openai import ChatOpenAI
        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gemini-2.5-flash",
                openai_api_key=api_key,
                openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                temperature=0
            )
        )
    else:
        from langchain_openai import ChatOpenAI
        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from langchain_openai import OpenAIEmbeddings
        if provider == "gemini":
            ragas_emb = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model="text-embedding-004",
                    openai_api_key=api_key,
                    openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            )
        else:
            ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        )

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        ) for s in samples
    ])

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    df = result.to_pandas()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if key in df.columns:
            val = df[key].mean()
            scores[key] = round(float(val), 4) if not pd.isna(val) else None
        else:
            scores[key] = None
    scores["per_question"] = df.to_dict(orient="records")
    return scores


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_evaluation(provider: str, api_key: str,
                   progress_fn: Optional[Callable] = None) -> dict:
    EVAL_STORE.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"  {msg}")
        if progress_fn: progress_fn(msg)

    # 1. Setup
    log("Setting up GraphRAG pipeline...")
    grag = setup_graphrag(lambda p: call_llm(p, provider, api_key, 300), log)

    # 2. Ground truths
    log("Preparing ground truth answers...")
    gt = get_ground_truths(grag, provider, api_key, log)

    # 3. Run all 3 modes on all 10 questions
    raw = {"local": [], "global": [], "hybrid": []}
    for i, q in enumerate(TEST_QUESTIONS):
        for mode in ["local", "global", "hybrid"]:
            log(f"[{i+1}/{len(TEST_QUESTIONS)}] {mode:6s} → {q['id']}")
            try:
                out = query_mode(grag, q["question"], mode, provider, api_key,
                                 ground_truth=gt.get(q["id"], ""))
                raw[mode].append({
                    "id": q["id"], "question": q["question"], "type": q["type"],
                    "answer": out["answer"], "contexts": out["contexts"],
                    "ground_truth": gt.get(q["id"], ""),
                    "graph_metrics": out.get("graph_metrics", {}),
                })
            except Exception as e:
                log(f"  ERROR {mode}/{q['id']}: {e}")
                raw[mode].append({
                    "id": q["id"], "question": q["question"], "type": q["type"],
                    "answer": f"[Error: {e}]", "contexts": [],
                    "ground_truth": gt.get(q["id"], ""),
                    "graph_metrics": {},
                })
            time.sleep(0.3)

    # 4. RAGAS scoring
    log("Running RAGAS scoring...")
    scored = {}
    for mode, samples in raw.items():
        log(f"  Scoring: {mode}...")
        try:
            scored[mode] = {"scores": score_with_ragas(samples, provider, api_key), "samples": samples}
        except Exception as e:
            log(f"  RAGAS failed for {mode}: {e}")
            scored[mode] = {"scores": {"error": str(e)}, "samples": samples}

    # 5. Save
    output = {
        "metadata": {
            "provider": provider,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transcripts": list(grag._ingested_sources.keys()) if hasattr(grag, "_ingested_sources") else [Path(p).stem for p in TRANSCRIPT_PATHS],
            "n_questions": len(TEST_QUESTIONS),
        },
        "questions": TEST_QUESTIONS,
        "results":   scored,
    }
    out_path = EVAL_STORE / "results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Results saved → {out_path}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "groq", "gemini"], default="gemini")
    parser.add_argument("--api_key",  default=None)
    args = parser.parse_args()

    key = args.api_key or (
        os.getenv("OPENAI_API_KEY") if args.provider == "openai" else (
            os.getenv("GEMINI_API_KEY") if args.provider == "gemini" else os.getenv("GROQ_API_KEY")
        )
    )
    if not key:
        print(f"Error: set --api_key or {args.provider.upper()}_API_KEY"); sys.exit(1)

    print(f"\nEvaluating GraphRAG | provider={args.provider} | {len(TEST_QUESTIONS)} questions\n")
    results = run_evaluation(args.provider, key)

    print("\n" + "=" * 62)
    print(f"{'Mode':<10} {'Faithful':>10} {'AnsRel':>8} {'CtxPrec':>9} {'CtxRec':>8}")
    print("=" * 62)
    for mode, data in results["results"].items():
        s = data["scores"]
        vals = [f"{s.get(m):.3f}" if s.get(m) is not None else "  N/A"
                for m in ["faithfulness","answer_relevancy","context_precision","context_recall"]]
        print(f"{mode:<10} {vals[0]:>10} {vals[1]:>8} {vals[2]:>9} {vals[3]:>8}")
    print("=" * 62)
    print("\nView dashboard:  streamlit run eval_app.py\n")
