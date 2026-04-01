# RAG vs GraphRAG — A Practical Comparison

Two fully working RAG systems built on the same podcast transcript corpus, designed to demonstrate **when and why a knowledge graph improves retrieval**.

| | Simple RAG | GraphRAG |
|--|-----------|---------|
| Retrieval | Hybrid dense + sparse | Hybrid + entity graph expansion |
| Query modes | 1 | 3 (Local · Global · Hybrid) |
| Entity awareness | No | Yes (spaCy NER + co-occurrence graph) |
| Cross-doc synthesis | Limited | Via community detection + LLM summaries |
| Evaluation | — | RAGAS + graph-specific metrics |
| Best questions | Factual look-up | Relational, thematic, cross-document |

---

## Repository Structure

```
.
├── Simple_RAG/
│   ├── rag_pipeline.py      # Chunking, embedding, BM25, RRF, reranking, generation
│   ├── app.py               # Streamlit chat UI
│   ├── requirements.txt
│   └── README.md
│
├── GraphRAG/
│   ├── graph_pipeline.py    # Extends Simple RAG + NER, graph, communities, 3 modes
│   ├── app.py               # Streamlit chat + Knowledge Graph inspector
│   ├── requirements.txt
│   └── README.md
│
├── evaluate.py              # RAGAS evaluation engine for GraphRAG's 3 modes
├── eval_app.py              # Streamlit evaluation dashboard
├── eval_requirements.txt    # Extra dependencies for evaluation
└── EVALUATION_GUIDE.md      # Step-by-step evaluation instructions
```

---

## How the Two Systems Differ

Both systems share the same ingestion foundation:

```
SRT file → parse → clean → chunk (500 tokens, 80 overlap)
        → BAAI/bge-small-en-v1.5 embeddings → ChromaDB
        → BM25 index
```

**Simple RAG** stops there and retrieves via RRF + cross-encoder reranking.

**GraphRAG** adds a second pass over every chunk:

```
Chunks → spaCy NER → entity co-occurrence graph (NetworkX)
       → community detection (greedy modularity)
       → LLM summary per community (cached)
```

At query time, GraphRAG can answer using three modes:

- **Local** — expands retrieval via the entity graph (1-hop neighbors)
- **Global** — routes the query to the most semantically matching communities
- **Hybrid** — combines both strategies

---

## Quick Start

### Simple RAG

```bash
cd Simple_RAG
pip install -r requirements.txt

# Create .env with your API key(s)
echo "OPENAI_API_KEY=sk-..."  > .env
# or
echo "GROQ_API_KEY=gsk_..." > .env

streamlit run app.py
```

### GraphRAG

```bash
cd GraphRAG
pip install -r requirements.txt
python -m spacy download en_core_web_sm

echo "OPENAI_API_KEY=sk-..." > .env

streamlit run app.py
```

---

## Evaluation

The evaluation framework benchmarks GraphRAG's three modes against 10 hand-crafted questions across three categories:

| Category | Question Focus | Expected Winner |
|----------|----------------|-----------------|
| Factual (3 Qs) | Single-source specific facts | Local mode |
| Relational (4 Qs) | Entity connections across docs | Local mode |
| Thematic (3 Qs) | Synthesis across all three episodes | Global mode |

### Metrics

**RAGAS** (LLM-as-judge):
- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Of all retrieved chunks, how many were relevant?
- **Context Recall** — Did retrieval surface all information needed?

**Graph-specific**:
- **Entity Coverage Rate** — % of ground-truth entities present in retrieved context
- **Graph Utilization** — Fraction of final chunks sourced from graph expansion
- **Community Coherence** — % of a community's top entities mentioned in its LLM summary

### Run evaluation

```bash
pip install -r eval_requirements.txt

# Run evaluation (saves results to eval_store/results.json)
python evaluate.py --provider openai --api_key sk-...
# or
python evaluate.py --provider groq --api_key gsk_...

# View dashboard
streamlit run eval_app.py
```

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for full instructions and interpretation guidance.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, 384-dim) |
| Vector DB | ChromaDB (HNSW, cosine similarity, persistent) |
| Sparse retrieval | BM25Okapi (`rank-bm25`) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| NER | spaCy `en_core_web_sm` |
| Graph | NetworkX |
| Community detection | Greedy modularity (NetworkX) |
| LLM | OpenAI `gpt-4o-mini` · Groq `llama-3.3-70b` |
| Evaluation | RAGAS |
| UI | Streamlit |

---

## When to Use Each System

**Use Simple RAG when:**
- Questions are straightforward and factual
- You need fast ingest and low latency
- The corpus is small or homogeneous

**Use GraphRAG when:**
- Questions involve relationships between people, organisations, or concepts
- You need thematic or cross-document synthesis
- You want to inspect *why* certain content was retrieved (entity tags, community labels)
- You're evaluating retrieval quality rigorously

---

## Sample Questions to Try

**Factual (Simple RAG excels)**
- "What did Jensen Huang say about the origins of CUDA?"
- "What is Sam Altman's view on open-sourcing GPT models?"

**Relational (GraphRAG Local excels)**
- "How are Elon Musk and OpenAI connected according to these episodes?"
- "What companies did Jensen Huang mention in the context of AI infrastructure?"

**Thematic (GraphRAG Global excels)**
- "What are the recurring AI safety themes across all three guests?"
- "How do Jensen, Sam, and Elon differ in their views on AGI timelines?"

---

## License

MIT
