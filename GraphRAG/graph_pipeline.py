"""
GraphRAG Pipeline - Knowledge Graph Enhanced RAG (Multi-Transcript)

Pipeline:
  SRT → Clean → Chunk → NER (spaCy) → Entity Co-occurrence Graph (NetworkX)
  → Community Detection (all components) → LLM Community Summaries
  → Local / Global / Hybrid Query Modes

Query modes:
  - Local:  Dense+BM25 + 1-hop entity graph expansion → richer context
  - Global: Semantic match against community summaries → thematic answer
  - Hybrid: Both combined
"""

import re
import os
import json
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 500))
_DEFAULT_OVERLAP    = int(os.getenv("RAG_OVERLAP",     80))
_DEFAULT_TOP_K      = int(os.getenv("RAG_TOP_K",       20))
_DEFAULT_TOP_N      = int(os.getenv("RAG_TOP_N",        5))

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SRTBlock:
    index: int
    start_time: str
    end_time: str
    text: str

@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start_time: str
    end_time: str
    char_start: int
    char_end: int
    source: str
    token_count: int = 0


# ─── SRT Helpers ──────────────────────────────────────────────────────────────

def parse_srt(file_path: str) -> List[SRTBlock]:
    """Parse .srt file into structured blocks, falling back to plain text parsing if not valid SRT."""
    content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    
    # Normalize Windows/Mac line endings to Unix style for consistent splitting
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    
    # Try parsing as standard SRT first
    raw_blocks = re.split(r"\n\s*\n", content.strip())
    blocks = []
    for raw in raw_blocks:
        lines = raw.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip()
        )
        if not m:
            continue
        blocks.append(SRTBlock(
            index=idx,
            start_time=m.group(1).replace(",", "."),
            end_time=m.group(2).replace(",", "."),
            text=" ".join(lines[2:]).strip()
        ))

    # If no valid SRT blocks were parsed, fallback to plain text parsing (e.g., for standard .txt transcripts)
    if not blocks:
        paragraphs = re.split(r"\n\s*\n", content.strip())
        for idx, para in enumerate(paragraphs, 1):
            text = para.strip()
            if len(text) < 3:
                continue
            blocks.append(SRTBlock(
                index=idx,
                start_time="00:00:00.000",
                end_time="00:00:00.000",
                text=text
            ))

    return blocks


def clean_srt_blocks(blocks: List[SRTBlock]) -> List[SRTBlock]:
    patterns = [r"\(Transcribed by.*?\)", r"\[.*?\]", r"♪.*?♪", r"<[^>]+>"]
    cleaned  = []
    for b in blocks:
        text = b.text
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3:
            continue
        cleaned.append(SRTBlock(index=b.index, start_time=b.start_time, end_time=b.end_time, text=text))
    return cleaned


def reconstruct_full_text(blocks: List[SRTBlock]) -> list:
    if not blocks:
        return []
    segments, buf, buf_start, buf_end = [], [], None, None
    for b in blocks:
        if buf_start is None:
            buf_start = b.start_time
        buf.append(b.text)
        buf_end = b.end_time
        if b.text and b.text[-1] in ".!?":
            segments.append({"text": " ".join(buf), "start_time": buf_start, "end_time": buf_end})
            buf, buf_start, buf_end = [], None, None
    if buf:
        segments.append({
            "text": " ".join(buf),
            "start_time": buf_start or blocks[0].start_time,
            "end_time":   buf_end   or blocks[-1].end_time
        })
    return segments


def chunk_segments(segments: list, chunk_size=500, overlap=80, source_name="podcast") -> List[TextChunk]:
    if not segments:
        return []
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count_tokens(t): return len(enc.encode(t))
    except Exception:
        def count_tokens(t): return len(t.split())

    all_text = " ".join(s["text"] for s in segments)
    words    = all_text.split()
    if not words:
        return []

    char_to_seg: Dict[int, int] = {}
    pos = 0
    for i, seg in enumerate(segments):
        for _ in range(len(seg["text"]) + 1):
            char_to_seg[pos] = i
            pos += 1
        pos += 1

    chunks, chunk_id, i = [], 0, 0
    while i < len(words):
        window, tc, j = [], 0, i
        while j < len(words) and tc < chunk_size:
            window.append(words[j])
            tc = count_tokens(" ".join(window))
            j += 1
        if not window:
            break
        chunk_text = " ".join(window)
        prefix     = " ".join(words[:i])
        csc        = len(prefix) + (1 if i > 0 else 0)
        n_segs     = len(segments)
        seg_s      = min(char_to_seg.get(csc, 0), n_segs - 1)
        seg_e      = min(char_to_seg.get(csc + len(chunk_text), 0), n_segs - 1)
        chunks.append(TextChunk(
            chunk_id=chunk_id, text=chunk_text,
            start_time=segments[seg_s]["start_time"],
            end_time=segments[seg_e]["end_time"],
            char_start=csc, char_end=csc + len(chunk_text),
            source=source_name, token_count=tc
        ))
        chunk_id += 1
        ow, ot, k = [], 0, j - 1
        while k > i and ot < overlap:
            ow.insert(0, words[k])
            ot = count_tokens(" ".join(ow))
            k -= 1
        i += max(1, j - i - len(ow))
    return chunks


# ─── Vector Store ─────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "graphrag_gemini"):
        import chromadb
        self.client          = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name

    def _get_api_key(self) -> str:
        # 1. Try env variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # 2. Try Streamlit session state if running inside dashboard
            try:
                import streamlit as st
                if "api_key" in st.session_state and st.session_state.api_key:
                    api_key = st.session_state.api_key
            except ImportError:
                pass
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable "
                "or configure it in your Streamlit dashboard sidebar."
            )
        return api_key

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        api_key = self._get_api_key()
        from google import genai
        client = genai.Client(api_key=api_key)
        
        # Sanitize whitespace/empty entries to prevent Gemini API errors
        sanitized = [t if (t and t.strip()) else " " for t in texts]
        
        # Batch requests in blocks of at most 100 to respect strict Gemini API constraints
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(sanitized), batch_size):
            batch = sanitized[i : i + batch_size]
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch
            )
            if response and response.embeddings:
                embeddings.extend([emb.values for emb in response.embeddings])
            else:
                raise ValueError("Gemini API call returned empty embeddings response.")
                
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        api_key = self._get_api_key()
        from google import genai
        client = genai.Client(api_key=api_key)
        
        sanitized = query if (query and query.strip()) else " "
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=sanitized
        )
        if response and response.embeddings:
            return response.embeddings[0].values
        raise ValueError("Gemini API query embedding call returned empty response.")

    def index_chunks(self, chunks: List[TextChunk], source_id: str):
        if not chunks:
            return None
        try:
            existing = self.client.get_collection(self.collection_name)
            results  = existing.get(where={"source": {"$eq": chunks[0].source}}, limit=1)
            if results["ids"]:
                print(f"Source '{source_id}' already in ChromaDB. Skipping embedding.")
                return existing
        except Exception:
            pass

        collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        texts      = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks for '{source_id}'...")
        embeddings = self.embed_texts(texts)
        ids        = [f"{source_id}_chunk_{c.chunk_id}" for c in chunks]
        metadatas  = [{
            "source": c.source, "chunk_id": c.chunk_id,
            "start_time": c.start_time, "end_time": c.end_time,
            "token_count": c.token_count
        } for c in chunks]
        for i in range(0, len(chunks), 100):
            collection.upsert(
                ids=ids[i:i+100], embeddings=embeddings[i:i+100],
                documents=texts[i:i+100], metadatas=metadatas[i:i+100]
            )
        print(f"✅ Indexed {len(chunks)} chunks.")
        return collection

    def dense_search(self, query: str, top_k: int = 20) -> List[Tuple[str, dict, float]]:
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return []
        count = collection.count()
        if count == 0:
            return []
        query_emb = self.embed_query(query)
        results   = collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )
        return [
            (doc, meta, 1.0 - dist)
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]


# ─── BM25 ──────────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self):
        self.bm25   = None
        self.chunks: List[TextChunk] = []

    def build(self, chunks: List[TextChunk]):
        if not chunks:
            return
        from rank_bm25 import BM25Okapi
        self.chunks   = list(chunks)
        self.bm25     = BM25Okapi([self._tok(c.text) for c in self.chunks])

    def _tok(self, text: str) -> List[str]:
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, dict, float]]:
        if self.bm25 is None or not self.chunks:
            return []
        scores  = self.bm25.get_scores(self._tok(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            (self.chunks[i].text,
             {"source": self.chunks[i].source, "chunk_id": self.chunks[i].chunk_id,
              "start_time": self.chunks[i].start_time, "end_time": self.chunks[i].end_time,
              "token_count": self.chunks[i].token_count},
             float(scores[i]))
            for i in top_idx if scores[i] > 0
        ]


# ─── RRF ───────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(dense, sparse, k=60, dw=0.6, sw=0.4):
    scores, docs = {}, {}
    for rank, (text, meta, _) in enumerate(dense):
        cid = f"{meta.get('source', '')}_{meta.get('chunk_id', rank)}"
        scores[cid] = scores.get(cid, 0.0) + dw / (k + rank + 1)
        docs[cid]   = (text, meta)
    for rank, (text, meta, _) in enumerate(sparse):
        cid = f"{meta.get('source', '')}_{meta.get('chunk_id', rank)}"
        scores[cid] = scores.get(cid, 0.0) + sw / (k + rank + 1)
        docs[cid]   = (text, meta)
    return [(docs[c][0], docs[c][1], scores[c])
            for c in sorted(scores, key=lambda x: scores[x], reverse=True)]


# ─── Reranker ──────────────────────────────────────────────────────────────────

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model    = None
        self.model_name = model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print("Loading reranker...")
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query, candidates, top_n=5):
        if not candidates:
            return []
        self._load()
        scores = self._model.predict([(query, t) for t, _, _ in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [(t, m, float(s)) for s, (t, m, _) in ranked[:top_n]]


# ─── Entity labels we care about in podcast transcripts ───────────────────────
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "LAW", "FAC"}


# ─── GraphRAG Core ─────────────────────────────────────────────────────────────

class GraphRAG:
    def __init__(
        self,
        persist_dir: str = "./graph_store",
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
        top_k_retrieve: int = _DEFAULT_TOP_K,
        top_n_rerank: int = _DEFAULT_TOP_N
    ):
        import networkx as nx

        self.persist_dir    = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        self.chunk_size     = chunk_size
        self.overlap        = overlap
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank   = top_n_rerank

        print(f"📐 GraphRAG config → chunk_size={chunk_size}, overlap={overlap}, "
              f"top_k={top_k_retrieve}, top_n={top_n_rerank}")

        self.vector_store = VectorStore(
            persist_dir=str(self.persist_dir / "chroma"), collection_name="graphrag_gemini"
        )
        self.bm25_index   = BM25Index()
        self.reranker     = Reranker()

        self.graph: nx.Graph                 = nx.Graph()
        self._entity_to_chunks: Dict[str, set] = {}   # entity → set of "source_chunkid" keys
        self._communities: Dict[int, dict]     = {}
        self._community_summary_embeddings: Optional[np.ndarray] = None
        self._community_ids_ordered: List[int] = []

        self._all_chunks: List[TextChunk]       = []
        self._ingested_sources: Dict[str, dict] = {}

        self._nlp = None

    # ── spaCy ──────────────────────────────────────────────────────────────────

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy en_core_web_sm loaded.")
            except OSError:
                raise RuntimeError(
                    "spaCy model missing. Run: python -m spacy download en_core_web_sm"
                )
        return self._nlp

    def extract_entities(self, text: str) -> List[str]:
        """Return unique lowercase entity strings from text."""
        nlp = self._get_nlp()
        doc = nlp(text)
        seen, entities = set(), []
        for ent in doc.ents:
            if ent.label_ in _ENTITY_LABELS:
                name = ent.text.strip().lower()
                # Filter very short strings and pure numbers
                if len(name) > 2 and not re.match(r"^\d+[\d\s,\.]*$", name):
                    if name not in seen:
                        seen.add(name)
                        entities.append(name)
        return entities

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, file_path: str, llm_fn: Optional[Callable] = None) -> dict:
        """
        Ingest one transcript:
          1. Parse/chunk SRT (cached per source)
          2. Extract entities → build co-occurrence graph
          3. Detect communities across all connected components
          4. Optionally generate community summaries via LLM
          5. Persist full state to disk
        """
        source_name = Path(file_path).stem

        if source_name in self._ingested_sources:
            print(f"⏭️  '{source_name}' already loaded this session.")
            return {**self._ingested_sources[source_name], "already_loaded": True}

        chunks_path = self.persist_dir / f"{self._hash(source_name)}_chunks.json"
        stats: dict = {"source": source_name}

        if chunks_path.exists():
            print(f"📦 Loading cached chunks for: {source_name}")
            source_chunks = self._load_source_chunks(chunks_path)
            stats["cached"] = True
        else:
            print(f"⚙️  Processing: {source_name}")
            blocks = parse_srt(file_path)
            stats["raw_blocks"]   = len(blocks)
            blocks = clean_srt_blocks(blocks)
            stats["clean_blocks"] = len(blocks)
            segments = reconstruct_full_text(blocks)
            stats["segments"]     = len(segments)
            source_chunks = chunk_segments(segments, self.chunk_size, self.overlap, source_name)
            stats["cached"] = False
            self._save_source_chunks(source_chunks, chunks_path)
            self.vector_store.index_chunks(source_chunks, self._hash(source_name))

        stats["chunks"] = len(source_chunks)
        self._all_chunks.extend(source_chunks)
        self._ingested_sources[source_name] = stats

        # Rebuild unified BM25
        print(f"🔄 Rebuilding BM25 over {len(self._all_chunks)} total chunks…")
        self.bm25_index.build(self._all_chunks)

        # Build / extend entity co-occurrence graph
        print("🕸️  Extracting entities and building graph…")
        self._build_graph(source_chunks)

        # Detect communities across ALL graph components
        print("🔍 Detecting communities…")
        self._detect_communities()

        if llm_fn:
            print("📝 Generating community summaries…")
            self.generate_summaries(llm_fn)

        stats.update({
            "entities":      self.graph.number_of_nodes(),
            "edges":         self.graph.number_of_edges(),
            "communities":   len(self._communities),
            "total_chunks":  len(self._all_chunks),
            "total_sources": len(self._ingested_sources),
        })
        self._ingested_sources[source_name] = stats

        # Persist full state so it survives page refresh
        self.save_state()

        return stats

    def ingest_multiple(self, file_paths: List[str], llm_fn=None) -> List[dict]:
        return [self.ingest(fp, llm_fn=llm_fn) for fp in file_paths]

    # ── Graph Construction ─────────────────────────────────────────────────────

    def _build_graph(self, chunks: List[TextChunk]):
        """Extract entities from chunks and add nodes/edges to the co-occurrence graph."""
        for chunk in tqdm(chunks, desc="Building entity graph", leave=False):
            entities = self.extract_entities(chunk.text)
            if not entities:
                continue

            chunk_key = f"{chunk.source}_{chunk.chunk_id}"
            for ent in entities:
                if not self.graph.has_node(ent):
                    self.graph.add_node(ent, count=0)
                self.graph.nodes[ent]["count"] += 1
                if ent not in self._entity_to_chunks:
                    self._entity_to_chunks[ent] = set()
                self._entity_to_chunks[ent].add(chunk_key)

            # Co-occurrence edges within the same chunk
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    if self.graph.has_edge(e1, e2):
                        self.graph[e1][e2]["weight"] += 1
                    else:
                        self.graph.add_edge(e1, e2, weight=1)

    # ── Community Detection (all components) ──────────────────────────────────

    def _detect_communities(self):
        """
        Run greedy modularity community detection on EVERY connected component.
        Singleton nodes each form their own community.
        """
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities

        if self.graph.number_of_nodes() == 0:
            return

        all_communities: List[frozenset] = []

        for component in nx.connected_components(self.graph):
            subgraph = self.graph.subgraph(component)
            if subgraph.number_of_nodes() == 1:
                all_communities.append(frozenset(component))
            else:
                try:
                    comms = greedy_modularity_communities(subgraph, weight="weight")
                    all_communities.extend(comms)
                except Exception:
                    # Fallback: treat whole component as one community
                    all_communities.append(frozenset(component))

        # Build community registry
        self._communities = {}
        for comm_id, members in enumerate(all_communities):
            members_sorted = sorted(
                members,
                key=lambda e: self.graph.nodes[e].get("count", 0),
                reverse=True
            )
            chunk_keys: set = set()
            for ent in members_sorted:
                chunk_keys.update(self._entity_to_chunks.get(ent, set()))

            self._communities[comm_id] = {
                "entities":   members_sorted,
                "chunk_keys": list(chunk_keys),
                "summary":    None
            }

        # Sort by community size (largest first)
        self._communities = dict(
            sorted(self._communities.items(), key=lambda x: -len(x[1]["entities"]))
        )
        # Reassign sequential IDs after sort
        self._communities = {i: v for i, v in enumerate(self._communities.values())}

        # Invalidate cached summary embeddings — graph changed
        self._community_summary_embeddings = None
        self._community_ids_ordered        = []

    # ── Community Summaries ────────────────────────────────────────────────────

    def generate_summaries(self, llm_fn: Callable, force: bool = False):
        """Generate a 2-3 sentence LLM summary for each community."""
        chunk_lookup = self._make_chunk_lookup()
        generated    = 0

        for comm_id, comm_data in self._communities.items():
            if comm_data["summary"] and not force:
                continue
            # Skip tiny singletons — not informative
            if len(comm_data["entities"]) < 2:
                comm_data["summary"] = f"Single entity cluster: {comm_data['entities'][0] if comm_data['entities'] else '—'}"
                continue

            top_entities = comm_data["entities"][:15]
            sample_keys  = comm_data["chunk_keys"][:6]
            sample_texts = [chunk_lookup[k].text for k in sample_keys if k in chunk_lookup]
            context      = "\n\n".join(sample_texts[:5]) if sample_texts else "(no text available)"

            prompt = (
                f"You are analyzing a cluster of entities extracted from podcast transcripts.\n\n"
                f"Key entities in this cluster: {', '.join(top_entities)}\n\n"
                f"Relevant transcript excerpts:\n{context}\n\n"
                f"Write a 2-3 sentence summary describing the topic/theme this cluster represents. "
                f"Be specific and grounded in the transcript content."
            )
            try:
                summary = llm_fn(prompt)
            except Exception as e:
                summary = f"Cluster covering: {', '.join(top_entities[:5])} (auto-summary unavailable: {e})"

            comm_data["summary"] = summary
            generated += 1

        print(f"✅ Generated {generated} community summaries.")
        self._cache_summary_embeddings()
        # Persist updated state
        self.save_state()

    def _cache_summary_embeddings(self):
        """Pre-embed all community summaries for fast semantic search in global mode."""
        summaries = [
            (cid, cd["summary"])
            for cid, cd in self._communities.items()
            if cd.get("summary") and len(cd["entities"]) >= 2
        ]
        if not summaries:
            self._community_summary_embeddings = None
            self._community_ids_ordered        = []
            return

        ids   = [s[0] for s in summaries]
        texts = [s[1] for s in summaries]
        embs  = self.vector_store.embed_texts(texts)

        self._community_ids_ordered        = ids
        self._community_summary_embeddings = np.array(embs)

    # ── Query Methods ──────────────────────────────────────────────────────────

    def query(self, question: str, mode: str = "hybrid") -> dict:
        """
        Main query entry point.

        Returns:
            hits      - list of (text, meta, score) tuples
            mode      - actual mode used
            summaries - list of community summary strings used (may be empty)
            entities  - set of entities involved in the query path
        """
        if mode == "local":
            return self._local_query(question)
        elif mode == "global":
            return self._global_query(question)
        else:
            return self._hybrid_query(question)

    def _local_query(self, question: str) -> dict:
        """
        Graph-augmented local retrieval:
          Dense+BM25 → top-chunk entities → 1-hop graph neighbors → expanded context → rerank
        """
        # Standard retrieval
        dense  = self.vector_store.dense_search(question, top_k=self.top_k_retrieve)
        sparse = self.bm25_index.search(question, top_k=self.top_k_retrieve)
        fused  = reciprocal_rank_fusion(dense, sparse)

        # Entities from query
        query_ents = set(self.extract_entities(question))

        # Entities from top-5 retrieved chunks
        chunk_ents: set = set()
        for text, _, _ in fused[:5]:
            chunk_ents.update(self.extract_entities(text))

        seed_ents = query_ents | chunk_ents

        # 1-hop graph expansion — top neighbors by edge weight
        neighbor_ents: set = set()
        for ent in seed_ents:
            if ent in self.graph:
                nbrs = sorted(
                    self.graph.neighbors(ent),
                    key=lambda n: self.graph[ent][n].get("weight", 1),
                    reverse=True
                )
                neighbor_ents.update(nbrs[:5])

        # Collect chunks from neighbor entities
        chunk_lookup  = self._make_chunk_lookup()
        extra_keys: set = set()
        for ent in neighbor_ents:
            extra_keys.update(self._entity_to_chunks.get(ent, set()))

        extra_hits = []
        for key in list(extra_keys)[:20]:
            c = chunk_lookup.get(key)
            if c:
                extra_hits.append((
                    c.text,
                    {"source": c.source, "chunk_id": c.chunk_id,
                     "start_time": c.start_time, "end_time": c.end_time,
                     "token_count": c.token_count},
                    0.4   # baseline score for graph-expanded chunks
                ))

        combined = self._dedup(fused + extra_hits)
        reranked  = self.reranker.rerank(question, combined[:25], top_n=self.top_n_rerank)

        return {
            "hits":      reranked,
            "mode":      "local",
            "summaries": [],
            "entities":  sorted(seed_ents | neighbor_ents)
        }

    def _global_query(self, question: str) -> dict:
        """
        Community-summary-based retrieval:
          Embed query → cosine sim vs community summaries → top 3 communities → context
        """
        if (self._community_summary_embeddings is None
                or len(self._community_ids_ordered) == 0):
            # No summaries yet — fall back to local retrieval
            result = self._local_query(question)
            result["mode"] = "global_fallback"
            return result

        query_emb = np.array(self.vector_store.embed_query(question))
        sims      = self._community_summary_embeddings @ query_emb  # (n_communities,)
        top_idx   = np.argsort(sims)[::-1][:3]

        chunk_lookup       = self._make_chunk_lookup()
        selected_hits      = []
        selected_summaries = []

        for idx in top_idx:
            cid = self._community_ids_ordered[idx]
            cd  = self._communities.get(cid, {})
            if not cd:
                continue
            selected_summaries.append(cd["summary"])
            for key in cd["chunk_keys"][:6]:
                c = chunk_lookup.get(key)
                if c:
                    selected_hits.append((
                        c.text,
                        {"source": c.source, "chunk_id": c.chunk_id,
                         "start_time": c.start_time, "end_time": c.end_time,
                         "token_count": c.token_count},
                        float(sims[idx])
                    ))

        selected_hits = self._dedup(selected_hits)
        if len(selected_hits) >= 2:
            reranked = self.reranker.rerank(question, selected_hits, top_n=self.top_n_rerank)
        else:
            reranked = selected_hits[:self.top_n_rerank]

        return {
            "hits":      reranked,
            "mode":      "global",
            "summaries": selected_summaries,
            "entities":  []
        }

    def _hybrid_query(self, question: str) -> dict:
        """Combine local graph-expanded retrieval with global community context."""
        local_result  = self._local_query(question)
        global_result = self._global_query(question)

        seen: set = set()
        combined  = []
        for text, meta, score in local_result["hits"]:
            key = f"{meta.get('source', '')}_{meta.get('chunk_id', '')}"
            seen.add(key)
            combined.append((text, meta, score))

        for text, meta, score in global_result["hits"]:
            key = f"{meta.get('source', '')}_{meta.get('chunk_id', '')}"
            if key not in seen:
                seen.add(key)
                combined.append((text, meta, score * 0.8))

        reranked = self.reranker.rerank(question, combined[:25], top_n=self.top_n_rerank)

        return {
            "hits":      reranked,
            "mode":      "hybrid",
            "summaries": global_result["summaries"],
            "entities":  local_result["entities"]
        }

    def format_context(self, result: dict) -> str:
        """Format query result into LLM context string."""
        parts: List[str] = []
        summaries = result.get("summaries", [])

        if summaries:
            header = "## Relevant Topic/Theme Summaries (Knowledge Graph Communities)\n"
            header += "\n\n".join(f"• {s}" for s in summaries)
            parts.append(header)

        hits = result.get("hits", [])
        if not hits:
            parts.append("No specific excerpts retrieved.")
        else:
            for i, (text, meta, score) in enumerate(hits, 1):
                source    = meta.get("source", "unknown")
                timestamp = f"{meta.get('start_time', '?')} → {meta.get('end_time', '?')}"
                parts.append(f"[Excerpt {i} | Source: {source} | Timestamp: {timestamp}]\n{text}")

        return "\n\n---\n\n".join(parts) if parts else "No relevant content found."

    # ── Graph Analytics ────────────────────────────────────────────────────────

    def get_graph_stats(self) -> dict:
        import networkx as nx
        n = self.graph.number_of_nodes()
        return {
            "nodes":       n,
            "edges":       self.graph.number_of_edges(),
            "communities": len(self._communities),
            "density":     round(nx.density(self.graph), 4) if n > 1 else 0.0,
        }

    def get_top_entities(self, n: int = 25) -> List[Tuple[str, int]]:
        return sorted(
            [(node, data.get("count", 0)) for node, data in self.graph.nodes(data=True)],
            key=lambda x: -x[1]
        )[:n]

    def get_community_list(self) -> List[dict]:
        return [
            {
                "id":           cid,
                "size":         len(cd["entities"]),
                "top_entities": cd["entities"][:7],
                "summary":      cd.get("summary") or "No summary yet.",
                "num_chunks":   len(cd["chunk_keys"])
            }
            for cid, cd in self._communities.items()
        ]

    def has_summaries(self) -> bool:
        return any(
            cd.get("summary") for cd in self._communities.values()
            if len(cd.get("entities", [])) >= 2
        )

    # ── State Persistence ──────────────────────────────────────────────────────

    def save_state(self):
        """Save graph, entity mappings, communities, and ingested sources to disk."""
        state = {
            "graph":            self.graph,
            "entity_to_chunks": self._entity_to_chunks,
            "communities":      self._communities,
            "ingested_sources": self._ingested_sources,
        }
        path = self.persist_dir / "graph_state.pkl"
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"✅ Graph state saved ({self.graph.number_of_nodes()} nodes, "
              f"{len(self._communities)} communities).")

    def load_state(self) -> bool:
        """
        Restore graph state from disk. Returns True if successful.
        Also rebuilds BM25 from cached chunk JSON files.
        """
        path = self.persist_dir / "graph_state.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self.graph                = state["graph"]
            self._entity_to_chunks    = state["entity_to_chunks"]
            self._communities         = state["communities"]
            self._ingested_sources    = state.get("ingested_sources", {})

            # Rebuild _all_chunks + BM25 from per-source JSON cache
            self._rebuild_chunks_from_cache()

            # Rebuild community summary embeddings if summaries exist
            if self.has_summaries():
                self._cache_summary_embeddings()

            print(f"✅ Graph state restored: {self.graph.number_of_nodes()} entities, "
                  f"{len(self._ingested_sources)} sources, "
                  f"{len(self._communities)} communities.")
            return True

        except Exception as e:
            print(f"⚠️  Could not restore graph state: {e}")
            return False

    def _rebuild_chunks_from_cache(self):
        """Reload all source chunks from disk for BM25 rebuild."""
        self._all_chunks = []
        for source_name in self._ingested_sources:
            chunks_path = self.persist_dir / f"{self._hash(source_name)}_chunks.json"
            if chunks_path.exists():
                self._all_chunks.extend(self._load_source_chunks(chunks_path))
        if self._all_chunks:
            self.bm25_index.build(self._all_chunks)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _hash(self, name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()[:10]

    def _make_chunk_lookup(self) -> Dict[str, TextChunk]:
        return {f"{c.source}_{c.chunk_id}": c for c in self._all_chunks}

    def _dedup(self, hits: List[Tuple]) -> List[Tuple]:
        seen, out = set(), []
        for text, meta, score in hits:
            key = f"{meta.get('source', '')}_{meta.get('chunk_id', '')}"
            if key not in seen:
                seen.add(key)
                out.append((text, meta, score))
        return out

    def _save_source_chunks(self, chunks: List[TextChunk], path: Path):
        data = [
            {"chunk_id": c.chunk_id, "text": c.text, "start_time": c.start_time,
             "end_time": c.end_time, "char_start": c.char_start, "char_end": c.char_end,
             "source": c.source, "token_count": c.token_count}
            for c in chunks
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load_source_chunks(self, path: Path) -> List[TextChunk]:
        return [TextChunk(**d) for d in json.loads(path.read_text())]
