"""Index construction: dense vectors (NVIDIA) + sparse BM25.

**On not using a vector database.** At this corpus size -- a few thousand chunks,
2048 dimensions -- a flat numpy matrix with a normalized dot product is an *exact*
nearest-neighbour search that completes in single-digit milliseconds. An ANN index
(FAISS/LanceDB/pgvector) would add a dependency, a build step, and approximation
error to solve a problem we do not have. The honest engineering call at this scale
is a flat index; the switch to ANN belongs at ~1M vectors, and the retrieval
interface here is deliberately narrow so that swap stays a one-file change.

Dense and sparse are built from different text:
  dense  -> chunk.embed_text (breadcrumb + content, so the embedder gets context)
  sparse -> chunk.text       (raw content; the breadcrumb would skew term stats,
                              since every chunk in a filing shares its prefix)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import bm25s
import numpy as np

from ..ingest.chunk import Chunk
from ..nvidia import EMBED_NEMOTRON_3, embed


@dataclass
class Index:
    chunks: list[dict[str, Any]]
    vectors: np.ndarray          # (n, dim) float32, L2-normalized
    embed_model: str
    retriever: Any               # bm25s.BM25
    corpus_tokens: Any

    def __len__(self) -> int:
        return len(self.chunks)


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product is cosine similarity.

    Also required if you ever slice these vectors for Matryoshka dimensionality
    reduction -- MRL slicing invalidates the original norm.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def build(
    chunks: Sequence[Chunk],
    *,
    out_dir: Path = Path("data/processed"),
    embed_model: str = EMBED_NEMOTRON_3.id,
    batch_size: int = 32,
    verbose: bool = True,
) -> Index:
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"embedding {len(chunks)} chunks via {embed_model} ...")
    # input_type="passage": these embedders are asymmetric. Encoding documents
    # with the query prefix measurably degrades retrieval and raises no error.
    vecs = embed(
        [c.embed_text for c in chunks],
        model=embed_model,
        input_type="passage",
        batch_size=batch_size,
    )
    vectors = _normalize(np.asarray(vecs, dtype=np.float32))

    if verbose:
        print(f"building BM25 over {len(chunks)} chunks ...")
    # Index the same representation dense gets (breadcrumb + text). The first
    # release indexed bare text, which meant BM25 could not see the ticker,
    # form, year or item -- and 0 of the eval's gold chunks contain the company
    # name in their raw text while every question opens with it. The retrieval
    # ablation was comparing dense-with-metadata against sparse-without, and
    # part of "BM25 collapses on this corpus" was that asymmetry, not tables.
    corpus = [c.embed_text for c in chunks]
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    payload = [c.to_dict() for c in chunks]
    (out_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in payload), encoding="utf-8"
    )
    np.save(out_dir / "vectors.npy", vectors)
    (out_dir / "meta.json").write_text(
        json.dumps({"embed_model": embed_model, "dim": int(vectors.shape[1]),
                    "n_chunks": len(chunks)}, indent=2)
    )
    # bm25s' native save writes numpy arrays + JSON vocab -- no pickle, so
    # loading an index does not execute arbitrary code.
    retriever.save(str(out_dir / "bm25"), corpus=None)

    if verbose:
        print(f"index written to {out_dir}  ({len(chunks)} chunks, dim={vectors.shape[1]})")
    return Index(payload, vectors, embed_model, retriever, corpus_tokens)


def load(in_dir: Path = Path("data/processed")) -> Index:
    meta = json.loads((in_dir / "meta.json").read_text())
    chunks = [
        json.loads(l)
        for l in (in_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if l
    ]
    vectors = np.load(in_dir / "vectors.npy")
    retriever = bm25s.BM25.load(str(in_dir / "bm25"), load_corpus=False)
    # Re-tokenize rather than persisting tokens: it is fast at this scale and
    # keeps the on-disk format pickle-free.
    corpus_tokens = bm25s.tokenize(
        [c["text"] for c in chunks], stopwords="en", show_progress=False
    )
    return Index(chunks, vectors, meta["embed_model"], retriever, corpus_tokens)
