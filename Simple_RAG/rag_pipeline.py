"""
Podcast RAG Pipeline - Core Module (Multi-Transcript)
Handles: SRT parsing → Cleaning → Chunking → Embedding → BM25 → Hybrid Retrieval → Reranking
Supports multiple transcript files and session-restore from disk cache.
"""

import re
import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─── Env-Driven Defaults ──────────────────────────────────────────────────────
_DEFAULT_CHUNK_SIZE  = int(os.getenv("RAG_CHUNK_SIZE", 500))
_DEFAULT_OVERLAP     = int(os.getenv("RAG_OVERLAP",     80))
_DEFAULT_TOP_K       = int(os.getenv("RAG_TOP_K",       20))
_DEFAULT_TOP_N       = int(os.getenv("RAG_TOP_N",        5))

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

# ─── Phase 1: SRT Parser ──────────────────────────────────────────────────────

def parse_srt(file_path: str) -> List[SRTBlock]:
    """Parse .srt file into structured blocks."""
    content = Path(file_path).read_text(encoding="utf-8", errors="replace")
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

        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip()
        )
        if not time_match:
            continue

        start_time = time_match.group(1).replace(",", ".")
        end_time   = time_match.group(2).replace(",", ".")
        text = " ".join(lines[2:]).strip()
        blocks.append(SRTBlock(index=idx, start_time=start_time, end_time=end_time, text=text))

    return blocks


# ─── Phase 2: Cleaning & Reconstruction ───────────────────────────────────────

def clean_srt_blocks(blocks: List[SRTBlock]) -> List[SRTBlock]:
    """Clean noise from auto-generated transcripts."""
    noise_patterns = [
        r"\(Transcribed by.*?\)",
        r"\[.*?\]",
        r"♪.*?♪",
        r"<[^>]+>",
    ]
    cleaned = []
    for block in blocks:
        text = block.text
        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3:
            continue
        cleaned.append(SRTBlock(
            index=block.index, start_time=block.start_time,
            end_time=block.end_time, text=text
        ))
    return cleaned


def reconstruct_full_text(blocks: List[SRTBlock]) -> list:
    """Merge SRT blocks into continuous sentence segments."""
    if not blocks:
        return []

    segments = []
    buffer: List[str] = []
    buffer_start: Optional[str] = None
    buffer_end: Optional[str] = None

    for block in blocks:
        text = block.text
        if buffer_start is None:
            buffer_start = block.start_time
        buffer.append(text)
        buffer_end = block.end_time

        if text and text[-1] in ".!?":
            segments.append({
                "text":       " ".join(buffer),
                "start_time": buffer_start,
                "end_time":   buffer_end
            })
            buffer, buffer_start, buffer_end = [], None, None

    if buffer:
        segments.append({
            "text":       " ".join(buffer),
            "start_time": buffer_start or blocks[0].start_time,
            "end_time":   buffer_end   or blocks[-1].end_time
        })

    return segments


# ─── Phase 3: Chunking ────────────────────────────────────────────────────────

def chunk_segments(
    segments: list,
    chunk_size: int = 500,
    overlap: int = 80,
    source_name: str = "podcast"
) -> List[TextChunk]:
    """Sliding window chunking over reconstructed segments."""
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

    # Build a char→segment_index lookup for timestamp resolution
    char_to_seg: Dict[int, int] = {}
    pos = 0
    for i, seg in enumerate(segments):
        for _ in range(len(seg["text"]) + 1):
            char_to_seg[pos] = i
            pos += 1
        pos += 1  # space between segments

    chunks: List[TextChunk] = []
    chunk_id = 0
    i = 0

    while i < len(words):
        window: List[str] = []
        token_count = 0
        j = i

        while j < len(words) and token_count < chunk_size:
            window.append(words[j])
            token_count = count_tokens(" ".join(window))
            j += 1

        if not window:
            break

        chunk_text = " ".join(window)

        # Char position of this chunk start in the full text
        prefix = " ".join(words[:i])
        chunk_start_char = len(prefix) + (1 if i > 0 else 0)
        chunk_end_char   = chunk_start_char + len(chunk_text)

        n_segs      = len(segments)
        seg_idx     = min(char_to_seg.get(chunk_start_char, 0), n_segs - 1)
        end_seg_idx = min(char_to_seg.get(chunk_end_char, 0),   n_segs - 1)

        chunks.append(TextChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start_time=segments[seg_idx]["start_time"],
            end_time=segments[end_seg_idx]["end_time"],
            char_start=chunk_start_char,
            char_end=chunk_end_char,
            source=source_name,
            token_count=token_count
        ))
        chunk_id += 1

        # Overlap: find how many trailing words give ~overlap tokens
        overlap_words: List[str] = []
        overlap_tokens = 0
        k = j - 1
        while k > i and overlap_tokens < overlap:
            overlap_words.insert(0, words[k])
            overlap_tokens = count_tokens(" ".join(overlap_words))
            k -= 1

        step = max(1, j - i - len(overlap_words))
        i += step

    return chunks


# ─── Phase 4: Vector Store (ChromaDB) ─────────────────────────────────────────

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "podcast_rag"):
        import chromadb
        self.client          = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self._embedder       = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
            self._embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._embedder

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model    = self._get_embedder()
        prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        return model.encode(prefixed, batch_size=32, show_progress_bar=True, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        model = self._get_embedder()
        return model.encode(
            f"Represent this question for searching relevant passages: {query}",
            normalize_embeddings=True
        ).tolist()

    def index_chunks(self, chunks: List[TextChunk], source_id: str):
        """Embed and upsert chunks into ChromaDB. Skips if source already indexed."""
        if not chunks:
            print(f"No chunks to index for source '{source_id}'.")
            return None

        # Check if this source is already in the collection
        try:
            existing  = self.client.get_collection(self.collection_name)
            results   = existing.get(
                where={"source": {"$eq": chunks[0].source}}, limit=1
            )
            if results["ids"]:
                print(f"Source '{source_id}' already indexed. Skipping embedding.")
                return existing
        except Exception:
            pass  # Collection doesn't exist yet — create it below

        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        texts      = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks for '{source_id}'...")
        embeddings = self.embed_texts(texts)

        ids       = [f"{source_id}_chunk_{c.chunk_id}" for c in chunks]
        metadatas = [{
            "source":      c.source,
            "chunk_id":    c.chunk_id,
            "start_time":  c.start_time,
            "end_time":    c.end_time,
            "token_count": c.token_count
        } for c in chunks]

        for i in range(0, len(chunks), 100):
            collection.upsert(
                ids=ids[i:i+100],
                embeddings=embeddings[i:i+100],
                documents=texts[i:i+100],
                metadatas=metadatas[i:i+100]
            )

        print(f"✅ Indexed {len(chunks)} chunks into ChromaDB.")
        return collection

    def dense_search(self, query: str, top_k: int = 20) -> List[Tuple[str, dict, float]]:
        """Dense vector search. Returns (text, metadata, score) tuples."""
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return []  # Collection not yet created

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


# ─── Phase 5: BM25 Sparse Retrieval ───────────────────────────────────────────

class BM25Index:
    def __init__(self):
        self.bm25   = None
        self.chunks: List[TextChunk] = []

    def build(self, chunks: List[TextChunk]):
        if not chunks:
            return
        from rank_bm25 import BM25Okapi
        self.chunks   = list(chunks)
        tokenized     = [self._tok(c.text) for c in self.chunks]
        self.bm25     = BM25Okapi(tokenized)
        print(f"✅ BM25 index built over {len(self.chunks)} chunks.")

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


# ─── Phase 6: Hybrid Retrieval with RRF ───────────────────────────────────────

def reciprocal_rank_fusion(
    dense_hits: List[Tuple[str, dict, float]],
    sparse_hits: List[Tuple[str, dict, float]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> List[Tuple[str, dict, float]]:
    """
    Reciprocal Rank Fusion over dense + sparse results.
    Uses composite key (source + chunk_id) to avoid cross-transcript collisions.
    """
    scores: Dict[str, float] = {}
    docs:   Dict[str, Tuple] = {}

    for rank, (text, meta, _) in enumerate(dense_hits):
        cid = f"{meta.get('source', '')}_{meta.get('chunk_id', rank)}"
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank + 1)
        docs[cid]   = (text, meta)

    for rank, (text, meta, _) in enumerate(sparse_hits):
        cid = f"{meta.get('source', '')}_{meta.get('chunk_id', rank)}"
        scores[cid] = scores.get(cid, 0.0) + sparse_weight / (k + rank + 1)
        docs[cid]   = (text, meta)

    return [
        (docs[cid][0], docs[cid][1], scores[cid])
        for cid in sorted(scores, key=lambda x: scores[x], reverse=True)
    ]


# ─── Phase 7: Cross-Encoder Reranking ─────────────────────────────────────────

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model     = None
        self.model_name = model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print("Loading reranker model...")
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, dict, float]],
        top_n: int = 5
    ) -> List[Tuple[str, dict, float]]:
        if not candidates:
            return []
        self._load()
        scores = self._model.predict([(query, text) for text, _, _ in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [(text, meta, float(score)) for score, (text, meta, _) in ranked[:top_n]]


# ─── Full Pipeline Orchestrator (Multi-Transcript) ────────────────────────────

class PodcastRAG:
    def __init__(
        self,
        persist_dir: str = "./rag_store",
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
        top_k_retrieve: int = _DEFAULT_TOP_K,
        top_n_rerank: int = _DEFAULT_TOP_N
    ):
        self.persist_dir    = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        self.chunk_size     = chunk_size
        self.overlap        = overlap
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank   = top_n_rerank

        print(f"📐 RAG config → chunk_size={chunk_size}, overlap={overlap}, "
              f"top_k={top_k_retrieve}, top_n={top_n_rerank}")

        self.vector_store = VectorStore(persist_dir=str(self.persist_dir / "chroma"))
        self.bm25_index   = BM25Index()
        self.reranker     = Reranker()

        self._all_chunks: List[TextChunk]     = []
        self._ingested_sources: Dict[str, dict] = {}

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, file_path: str) -> dict:
        """
        Ingest one SRT/TXT transcript. Accumulates chunks across calls.
        Returns stats dict. Skips re-ingestion in same session.
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

        # Rebuild unified BM25 over ALL accumulated chunks
        print(f"🔄 Rebuilding BM25 over {len(self._all_chunks)} total chunks "
              f"({len(self._ingested_sources)} source(s))...")
        self.bm25_index.build(self._all_chunks)

        stats["total_chunks"]  = len(self._all_chunks)
        stats["total_sources"] = len(self._ingested_sources)

        # Persist state so we can restore after page refresh
        self._save_state()

        return stats

    def ingest_multiple(self, file_paths: List[str]) -> List[dict]:
        return [self.ingest(fp) for fp in file_paths]

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Tuple[str, dict, float]]:
        """Hybrid retrieval + reranking across ALL ingested transcripts."""
        dense_hits  = self.vector_store.dense_search(query, top_k=self.top_k_retrieve)
        sparse_hits = self.bm25_index.search(query, top_k=self.top_k_retrieve)
        fused       = reciprocal_rank_fusion(dense_hits, sparse_hits)

        if not fused:
            return []

        reranked = self.reranker.rerank(query, fused[:20], top_n=self.top_n_rerank)
        return reranked

    def format_context(self, hits: List[Tuple[str, dict, float]]) -> str:
        """Format retrieved chunks into LLM context string."""
        if not hits:
            return "No relevant excerpts found."
        parts = []
        for i, (text, meta, score) in enumerate(hits, 1):
            source    = meta.get("source", "unknown")
            timestamp = f"{meta.get('start_time', '?')} → {meta.get('end_time', '?')}"
            parts.append(f"[Excerpt {i} | Source: {source} | Timestamp: {timestamp}]\n{text}")
        return "\n\n---\n\n".join(parts)

    # ── State Persistence (survive page refresh) ───────────────────────────────

    def _save_state(self):
        """Save the list of ingested source names so we can restore on next session."""
        state = {"ingested_sources": list(self._ingested_sources.keys())}
        (self.persist_dir / "rag_state.json").write_text(json.dumps(state))

    def load_state(self) -> bool:
        """
        Restore previously ingested sources from disk cache.
        Returns True if any sources were restored.
        """
        state_path = self.persist_dir / "rag_state.json"
        if not state_path.exists():
            return False
        try:
            state   = json.loads(state_path.read_text())
            sources = state.get("ingested_sources", [])

            for name in sources:
                chunks_path = self.persist_dir / f"{self._hash(name)}_chunks.json"
                if not chunks_path.exists():
                    continue
                source_chunks = self._load_source_chunks(chunks_path)
                self._all_chunks.extend(source_chunks)
                total = len(self._all_chunks)
                self._ingested_sources[name] = {
                    "source":        name,
                    "chunks":        len(source_chunks),
                    "cached":        True,
                    "total_chunks":  total,
                    "total_sources": 0  # updated below
                }

            if self._all_chunks:
                self.bm25_index.build(self._all_chunks)
                # Fix total_sources counts
                n = len(self._ingested_sources)
                for v in self._ingested_sources.values():
                    v["total_sources"] = n

            print(f"✅ Restored {len(self._ingested_sources)} source(s) from cache.")
            return bool(self._ingested_sources)

        except Exception as e:
            print(f"⚠️  Could not restore RAG state: {e}")
            return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _hash(self, name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()[:10]

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
