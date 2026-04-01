# Simple RAG — Hybrid Retrieval-Augmented Generation

A production-quality RAG pipeline that answers questions over podcast transcripts using **hybrid dense + sparse retrieval**, cross-encoder reranking, and your choice of LLM.

---

## Architecture

```
SRT Transcripts
      │
      ▼
  parse_srt()          ← Extract text + timestamps from .srt files
      │
  clean_srt_blocks()   ← Remove filler markers, HTML, music symbols
      │
  chunk_segments()     ← Sliding window (500 tokens, 80 overlap)
      │
      ├──► VectorStore (ChromaDB)    ← Dense embeddings (BAAI/bge-small-en-v1.5)
      └──► BM25Index                 ← Sparse keyword index
                │
                ▼
         Query at runtime
                │
      Dense (top-20) + Sparse (top-20)
                │
      reciprocal_rank_fusion()       ← 60% dense · 40% sparse
                │
      Cross-encoder Reranker         ← ms-marco-MiniLM-L-6-v2 → top-5
                │
                ▼
          LLM Generation             ← gpt-4o-mini  or  llama-3.3-70b
                │
                ▼
        Answer + Source Excerpts
```

### Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | Fast, high-quality, runs locally |
| Vector DB | ChromaDB with HNSW + cosine | Persistent, no server needed |
| Sparse search | BM25Okapi | Catches exact keyword matches dense search misses |
| Fusion | Reciprocal Rank Fusion | Combines rankings without score normalization |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision boost before LLM call |
| LLM | OpenAI / Groq | User-selectable at runtime |

---

## Project Structure

```
Simple_RAG/
├── rag_pipeline.py      # Core logic: parsing, chunking, retrieval, generation
├── app.py               # Streamlit chat interface
├── requirements.txt     # Python dependencies
└── rag_store/           # Auto-created: ChromaDB + chunk cache + state
    ├── chroma/
    ├── rag_state.json
    └── <hash>_chunks.json
```

---

## Setup

### 1. Install dependencies

```bash
cd Simple_RAG
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the `Simple_RAG/` directory:

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

1. **Upload transcripts** — Drag `.srt` files into the sidebar upload area. Progress bars show chunking and embedding status. Ingested files are cached and persist across sessions.

2. **Select your LLM** — Switch between OpenAI (`gpt-4o-mini`) and Groq (`llama-3.3-70b`) from the sidebar.

3. **Tune retrieval** *(optional)* — Adjust chunk size, overlap, top-K candidates, and top-N final chunks via sidebar sliders.

4. **Ask questions** — Type any question in the chat input. The app returns an answer with expandable source excerpts and timestamps.

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RAG_CHUNK_SIZE` | 500 | Tokens per chunk |
| `RAG_OVERLAP` | 80 | Overlap tokens between consecutive chunks |
| `RAG_TOP_K` | 20 | Candidates retrieved before reranking |
| `RAG_TOP_N` | 5 | Final chunks passed to LLM |

---

## Retrieval Pipeline (Detail)

**Hybrid Search**
- Dense: embed query with `bge-small` → cosine similarity in ChromaDB → top-20
- Sparse: tokenize query → BM25 scoring → top-20
- Fusion: RRF with `k=60` smoothing, 60 % dense weight

**Reranking**
- Build (query, chunk) pairs for all unique candidates
- Score with cross-encoder → keep top-N by relevance score

**Context Assembly**
- Prepend source file name and timestamp range to each chunk
- Concatenate top-N chunks into a single context block for the LLM

---

## State Persistence

The pipeline saves state after every ingest so you never re-process the same file:

| File | Contents |
|------|----------|
| `rag_store/chroma/` | ChromaDB on-disk index (embeddings + metadata) |
| `rag_store/rag_state.json` | List of ingested source files |
| `rag_store/<hash>_chunks.json` | Raw chunks per source (for BM25 rebuild) |

On startup, the app detects existing state and restores the BM25 index without re-embedding.

---

## Limitations

- Flat retrieval — all chunks are treated equally regardless of speaker or topic
- No cross-document entity linking or relationship modeling
- Weaker at synthesis questions that require reasoning across multiple topics
- spaCy NER not used here — see the **GraphRAG** folder for entity-aware retrieval
