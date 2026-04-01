# GraphRAG Evaluation — Why It Matters

## What the Evaluation Does

The evaluation runs **10 carefully designed questions** across GraphRAG's three query modes
(Local, Global, Hybrid) and scores each answer using **four RAGAS metrics** plus **three
graph-specific structural metrics**. Results are saved to `eval_store/results.json` and
visualized in a Streamlit dashboard.

```
python evaluate.py --provider openai --api_key sk-...
# or
streamlit run eval_app.py
```

---

## Part 1 — RAGAS Metrics

These measure answer and retrieval quality. They treat the system as a black box —
they don't know anything about graphs.

### 1. Faithfulness
> *Does the answer stick to what was actually retrieved, or is the model making things up?*

GraphRAG retrieves rich, structured context — entity graph expansions and community
summaries. Faithfulness checks whether the LLM's answer is **grounded in that context**
or whether it drifted into hallucination.

**Why it matters for GraphRAG specifically:** GraphRAG retrieves more context than Simple
RAG (neighbors, summaries, plus raw chunks). More context = more opportunity for the LLM
to get creative and invent details. A high Faithfulness score proves the extra retrieval
is being used correctly, not ignored in favor of parametric memory.

---

### 2. Answer Relevancy
> *Does the answer actually address what was asked?*

This metric is orthogonal to faithfulness — an answer can be 100% grounded in context but
still miss the point of the question.

**Why it matters:** Local mode expands via entity graph neighbors. Some of those neighbors
are closely related; others are one hop away and only loosely relevant. Answer Relevancy
exposes whether the model is synthesizing toward the question or just summarizing everything
retrieved. Low relevancy in Local mode signals that graph expansion is pulling in too much
noise.

---

### 3. Context Precision
> *Of all the chunks retrieved, what fraction were actually useful?*

This is the **precision side of the retrieval trade-off**. Local mode deliberately adds
extra chunks through 1-hop graph expansion. Some are directly useful; some are tangential.

**Why it matters:** You want to catch cases where GraphRAG retrieves a lot (high recall)
but much of it is junk (low precision). For factual questions, precision should be high —
the graph expansion should narrow onto the relevant entity neighborhood. For thematic
questions, precision may dip because Global summaries cover broad topics and some content
is only tangentially relevant.

---

### 4. Context Recall
> *Did retrieval surface ALL the information needed to fully answer the question?*

This is the **most revealing RAGAS metric for GraphRAG**.

**Why it matters per question type:**

| Question Type | What to expect | Why |
|---|---|---|
| **Factual** | All 3 modes similar | The answer lives in one chunk; any retrieval finds it |
| **Relational** | Local > Global | Entity graph expansion pulls in linked chunks that flat search misses |
| **Thematic** | Global > Local | Community summaries pre-synthesize themes across all 3 transcripts |

High recall in Local mode on relational questions is the **smoking gun** that proves
graph expansion adds value over keyword/embedding search alone.

---

## Part 2 — Graph-Specific Structural Metrics

These three metrics are unique to GraphRAG. RAGAS cannot measure them because they
inspect the graph structure directly — not just the quality of the final answer.

### 5. Entity Coverage Rate *(all modes)*
> *What % of the ground-truth entities actually appeared in the retrieved context?*

Extracts named entities (people, orgs, products, places) from the ground-truth answer
using spaCy, then checks how many of those same entities appear anywhere in the retrieved
chunks and summaries.

**Why it matters:** RAGAS Context Recall measures whether the *information* was present.
Entity Coverage measures whether the *right actors and concepts* were present. A low
entity coverage score on relational questions tells you the retrieval found relevant-sounding
text but missed key entities — a symptom of flat semantic search that doesn't follow
entity connections.

**What to look for:**
- Local and Hybrid should have higher entity coverage than Global on relational questions,
  because graph expansion explicitly targets entity neighborhoods.
- All modes should be similar on factual questions (single-entity answer, easy to hit).
- Global should lead on thematic questions because community summaries explicitly name
  the key entities in each theme cluster.

---

### 6. Graph Utilization *(Local and Hybrid only)*
> *What fraction of the final retrieved chunks came exclusively from graph expansion,
> not from base dense+BM25 retrieval?*

After each Local/Hybrid query, the evaluation reruns the base retrieval (dense + BM25 +
RRF) without graph expansion and records which chunk IDs it would have returned. Any
chunk in the final reranked answer that was NOT reachable by base retrieval = contributed
by graph expansion. Graph Utilization = `graph-only chunks / total final chunks`.

**Why it matters:** This is the only metric that directly measures whether the graph is
doing work. If graph utilization is 0% on relational questions, the graph isn't adding
anything beyond what embedding search already found. If it's 40%+, the graph is surfacing
genuinely new context that base retrieval would have missed entirely.

**What to look for:**
- Graph utilization should be **highest on relational questions** — those are the ones
  where entity connections bridge gaps that embedding similarity can't close.
- Low utilization on factual questions is expected and fine — the relevant chunk scores
  high on embedding similarity and base retrieval finds it without graph help.
- Near-zero utilization across all question types would indicate the graph isn't
  contributing, which would be a signal to investigate the entity extraction or
  graph construction.

---

### 7. Community Coherence *(Global and Hybrid only)*
> *What % of a community's top entities are explicitly mentioned in its own
> LLM-generated summary?*

For each community summary used in a query, the evaluation looks up the community's top
10 entities in the graph and checks what fraction of them appear as substrings in the
summary text. The score is averaged across all communities used in that query.

**Why it matters:** Community summaries are the backbone of Global mode. If a summary is
vague or generic — "this cluster discusses various AI topics" — it won't match well
against specific thematic queries and Global's recall will suffer. High community
coherence (>60%) means the LLM summaries are grounded in the actual entities in each
cluster and will surface reliably when those entities are relevant to a query.

**What to look for:**
- Coherence should be consistently high (>50%) for Global mode to work well.
- Low coherence (< 30%) signals that the LLM summary generation prompt is too generic,
  or that community detection is grouping unrelated entities together.
- Hybrid coherence mirrors Global coherence since it uses the same summaries.

---

## The Three Query Modes — What Each One Proves

### Local Mode
Retrieves via vector similarity, then performs **1-hop entity graph expansion** — pulling
in chunks connected to the retrieved entities via the knowledge graph.

- **Best for:** Relational questions, entity-connection questions, "how are X and Y linked?"
- **Key metrics:** Context Recall ↑ on relational · Graph Utilization ↑ on relational · Entity Coverage ↑ on relational

### Global Mode
Retrieves against **LLM-generated community summaries** — pre-built thematic descriptions
of entity clusters that span all three documents.

- **Best for:** Thematic synthesis, "what are the main themes?", "what do all guests agree on?"
- **Key metrics:** Context Recall ↑ on thematic · Community Coherence (quality gate for summaries)

### Hybrid Mode
Combines Local (entity expansion) and Global (community summaries) in a single query.

- **Best for:** Complex questions that need both specific entity grounding and thematic breadth
- **Key metrics:** Should score well across ALL types — no category collapse

---

## The 10 Test Questions — Why They're Designed This Way

| ID | Type | Tests |
|---|---|---|
| Q01–Q03 | Factual | Baseline — all modes should be equal here |
| Q04–Q07 | Relational | Graph expansion — Local's unique capability |
| Q08–Q10 | Thematic | Community summaries — Global's unique capability |

**Factual (Q01–Q03)** are the control group. Similar scores across all modes confirms the
metrics are calibrated — there's no graph advantage when a single chunk holds the answer.

**Relational (Q04–Q07)** are where graph structure separates from flat retrieval. Q04
asks about the connection between Tesla, Autopilot, neural networks, and Dojo — a web of
entities spread across multiple chunks. Local mode's graph traversal surfaces this web;
embedding similarity alone misses the indirect links.

**Thematic (Q08–Q10)** are where Global mode shines. "What are the recurring themes across
all three conversations?" requires a bird's-eye view of the corpus — exactly what
community summaries provide.

---

## Why Ground Truths Are Generated via Hybrid Oracle

Reference answers are generated using Hybrid mode rather than being hand-written:

1. **Practical baseline** — uses the best available retrieval, representing a well-grounded answer given this corpus.
2. **Scales automatically** — no manual labeling when the question set changes.
3. **Consistent scope** — RAGAS scores reflect retrieval quality differences between modes, not gaps in domain knowledge.

**Known limitation:** Hybrid oracle answers can slightly favor Hybrid mode in Context
Recall scoring. This is acceptable when the goal is comparing Local vs Global vs Hybrid
against each other.

---

## How to Read the Dashboard

### Overall RAGAS Scores (Grouped Bar Chart)
Four metrics × three modes. Look for high Faithfulness across all modes and the
Local/Global recall split described above.

### Mode Capability Profiles (Radar Chart)
Visual fingerprint per mode. Hybrid should have the largest consistent area; Local and
Global should show complementary spikes.

### Context Recall by Question Type
**The key RAGAS chart.** Shows Local leading on Relational and Global leading on Thematic.
This directly validates GraphRAG's architectural design.

### Graph-Specific Structural Metrics
Three charts RAGAS can't produce:
- **Entity Coverage Rate by question type** — did retrieval find the right actors?
- **Graph Utilization cards** — is the graph actually contributing to answers?
- **Community Coherence cards** — are the summaries grounded in their entity clusters?

### Question-by-Question Answers
Side-by-side answers from all three modes with ground truth reference. The qualitative
difference on Q04–Q07 (relational questions) is the most compelling visual argument for
graph-structured retrieval.

---

## The Single Most Important Thing to Show in the Video

**Graph Utilization on relational questions, paired with Context Recall.**

Graph Utilization proves the *mechanism* — the graph is pulling in chunks that base
retrieval would never have found. Context Recall proves the *outcome* — those extra chunks
contain the information needed to fully answer the question.

Together they tell a complete story: *the graph finds what embeddings miss, and that
missing information is exactly what makes relational answers complete.*
