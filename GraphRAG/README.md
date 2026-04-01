# GraphRAG — Knowledge Graph Enhanced Retrieval-Augmented Generation

Extends hybrid RAG with **entity extraction**, **co-occurrence knowledge graph**, **community detection**, and **three distinct query modes** — each optimized for a different question type.

---

## Architecture

```
SRT Transcripts
      │
      ▼
  parse → clean → chunk → embed          ← Same pipeline as Simple RAG
      │
      ├──► VectorStore (ChromaDB)
      ├──► BM25Index
      │
      └──► Entity Extraction (spaCy NER)
                │
                ▼
         Co-occurrence Graph (NetworkX)
         Nodes: unique entities
         Edges: shared chunk, weight = co-occurrence count
                │
                ▼
         Community Detection (greedy modularity)
                │
                ▼
         LLM Community Summaries (cached)
                │
                ▼
    ┌───────────┼───────────┐
    ▼           ▼           ▼
  LOCAL       GLOBAL     HYBRID
  mode        mode        mode
```

### Three Query Modes

| Mode | Best For | Retrieval Strategy |
|------|----------|--------------------|
| **Local** | Specific facts, entity relationships | Dense+BM25 → RRF → extract query entities → 1-hop graph expansion → rerank |
| **Global** | Themes, synthesis, "what topics..." | Embed query → cosine vs community summaries → collect community chunks → rerank |
| **Hybrid** | Balanced / unknown question type | Run Local + Global → merge (0.8× weight to global) → rerank |

---

## Project Structure

```
GraphRAG/
├── graph_pipeline.py    # Core logic: graph construction, community detection, 3 query modes
├── app.py               # Streamlit chat + Knowledge Graph viewer
├── requirements.txt     # Python dependencies
└── graph_store/         # Auto-created: ChromaDB + graph state + chunk cache
    ├── chroma/
    ├── graph_state.pkl
    └── <hash>_chunks.json
```

---

## Setup

### 1. Install dependencies

```bash
cd GraphRAG
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API keys

Create a `.env` file in the `GraphRAG/` directory:

```env
# Choose one or both
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

### 3. Launch the app

```bash
streamlit run app.py
```

---

## Usage

### Chat Tab

1. **Upload transcripts** — Drag `.srt` files into the sidebar. Ingestion runs the full pipeline: chunking → embedding → NER → graph construction → community detection → summary generation (LLM call per community, cached to disk).

2. **Select query mode** — Choose Local, Global, or Hybrid depending on your question type.

3. **Select your LLM** — OpenAI (`gpt-4o-mini`) or Groq (`llama-3.3-70b`).

4. **Ask questions** — The response includes the answer, the entities used to expand retrieval, the communities considered, and source excerpts.

### Knowledge Graph Tab

Inspect the graph that was built from your documents:

- Node count, edge count, connected component statistics
- Top entities by degree centrality
- Community list with member entity counts
- Per-community entity breakdown

---

## Pipeline Detail

### Entity Extraction

spaCy `en_core_web_sm` NER extracts the following entity types per chunk:

`PERSON · ORG · GPE · PRODUCT · EVENT · WORK_OF_ART · NORP · LAW · FAC`

Filters applied: length > 2 characters, non-numeric, lowercased and deduplicated.

### Graph Construction

```python
for each chunk:
    entities = extract_entities(chunk.text)
    for each pair (e1, e2) in entities:
        graph.add_edge(e1, e2, weight += 1)
```

Maintains a reverse index: `entity → list of chunk IDs` for fast graph-guided retrieval.

### Community Detection

NetworkX `greedy_modularity_communities` runs independently per connected component. No manual hyperparameter tuning — the algorithm optimises modularity automatically.

### Community Summaries

One LLM call per community generates a paragraph-length summary of the community's key topics. Summaries are cached to `graph_store/graph_state.pkl` and reused on restart.

---

## Query Mode Detail

### Local Mode

```
1. Dense search (ChromaDB) → top-20
2. BM25 search             → top-20
3. RRF fusion              → merged ranking
4. Extract entities from query (spaCy)
5. For each entity:  look up graph neighbors (1 hop)
6. Collect chunks associated with neighbor entities
7. Merge with step-3 results, deduplicate
8. Cross-encoder rerank → top-5
9. LLM generation with entity context
```

### Global Mode

```
1. Embed query (bge-small)
2. Cosine similarity vs all community summary embeddings → top-3 communities
3. Collect all chunks belonging to those communities
4. Cross-encoder rerank → top-5
5. LLM generation with community summaries prepended to context
```

### Hybrid Mode

```
1. Run Local  → local_chunks   (entity-expanded)
2. Run Global → global_chunks  (community-selected)
3. Merge:  unique(local_chunks ∪ global_chunks)
           weight global chunks × 0.8 in reranking
4. Cross-encoder rerank → top-5
5. LLM generation with both chunk types + community summaries
```

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Tokens per chunk |
| `CHUNK_OVERLAP` | 80 | Overlap tokens |
| `TOP_K` | 20 | Candidates before reranking |
| `TOP_N` | 5 | Final chunks for LLM |
| `COMMUNITY_TOP_K` | 3 | Communities selected in Global mode |
| `GLOBAL_WEIGHT` | 0.8 | Score multiplier for global chunks in Hybrid mode |

---

## State Persistence

| File | Contents |
|------|----------|
| `graph_store/chroma/` | ChromaDB vector index |
| `graph_store/graph_state.pkl` | NetworkX graph + entity→chunks map + community assignments + LLM summaries |
| `graph_store/<hash>_chunks.json` | Raw text chunks per source |

> **Note:** `graph_state.pkl` is tied to the exact Python environment. If you upgrade `networkx` or change the pipeline classes, delete this file and re-ingest.

---

## When to Use Each Mode

| Question Type | Example | Recommended Mode |
|---------------|---------|-----------------|
| Specific fact | "What did Jensen say about CUDA?" | Local |
| Entity relationship | "How are Sam Altman and OpenAI connected?" | Local |
| Thematic | "What are the main AI safety themes discussed?" | Global |
| Cross-document synthesis | "Compare how each guest views AGI timelines" | Global or Hybrid |
| Unknown / general | Any question when unsure | Hybrid |

---

## Comparison with Simple RAG

| Feature | Simple RAG | GraphRAG |
|---------|------------|----------|
| Retrieval | Hybrid dense+sparse | Hybrid + graph expansion |
| Entity linking | No | Yes (spaCy NER) |
| Cross-doc reasoning | Limited | Via community summaries |
| Query modes | One (hybrid search) | Three (Local / Global / Hybrid) |
| Ingest time | Fast | Slower (NER + graph + LLM summaries) |
| Best for | Factual lookup | Relational + thematic questions |

---

## Limitations

- NER quality depends on `en_core_web_sm` — a small model. Rare entities may be missed.
- `graph_state.pkl` is not forward-compatible across major library versions.
- Community summary generation adds one LLM call per community on first ingest (cached afterward).
- Not designed for concurrent multi-user ingestion (single pickle file).
