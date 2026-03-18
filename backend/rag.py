import os
import time
import hashlib
import gc  # Garbage collection
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Optional
import torch
import numpy as np

# ── CONFIG ────────────────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chromadb")
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast 22MB embedder
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"  # tiny fast reranker

# ── MEMORY OPTIMISATION ───────────────────────────
# Load models with CPU optimisations
print("Loading embed model (optimised)...")
embedder = SentenceTransformer(EMBED_MODEL, device='cpu')
embedder.max_seq_length = 256  # Limit sequence length

print("Loading rerank model (optimised)...")
reranker = CrossEncoder(RERANK_MODEL, max_length=128, device='cpu')  # Reduced max_length

# Force garbage collection
gc.collect()

# ── CHROMADB WITH CORRECT CONFIGURATION ───────────
try:
    # Try the new ChromaDB 0.5.0+ format
    chroma = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=chromadb.config.Settings(
            anonymized_telemetry=False
        )
    )
    print(f"✅ ChromaDB initialized at {CHROMA_PATH}")
except Exception as e:
    print(f"⚠️  ChromaDB init error: {e}")
    print("Trying alternative configuration...")
    
    # Fallback for older versions or different setups
    chroma = chromadb.Client(chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))

def get_collection(doc_id: str):
    """Each document gets its own collection"""
    safe = "doc_" + hashlib.md5(doc_id.encode()).hexdigest()[:12]
    return chroma.get_or_create_collection(
        name=safe,
        metadata={"hnsw:space": "cosine", "doc_id": doc_id}
    )

def list_collections() -> List[Dict]:
    """List all uploaded documents"""
    cols = chroma.list_collections()
    result = []
    for c in cols:
        try:
            col = chroma.get_collection(c.name)
            result.append({
                "collection_id": c.name,
                "doc_id": c.metadata.get("doc_id", c.name),
                "chunk_count": col.count(),
                "metadata": c.metadata
            })
        except:
            continue
    return result


# ── MEMORY-EFFICIENT CHUNKING ─────────────────────
def chunk_text_streaming(text: str, chunk_size: int = 50, overlap: int = 10) -> List[Dict]:
    """
    Stream-based chunking to avoid loading entire text into memory at once
    Smaller chunk size (50) and overlap (10) for memory efficiency
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_no = 0
    
    # Process in smaller batches
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        
        # Only store essential data
        chunks.append({
            "text": chunk,
            "chunk_no": chunk_no,
            "words": end - start
        })
        
        chunk_no += 1
        start = end - overlap
        
        # Clear temporary variables
        if chunk_no % 10 == 0:
            gc.collect()
    
    return chunks


# ── BATCH PROCESSING FOR EMBEDDINGS ───────────────
def batch_encode(texts: List[str], batch_size: int = 8) -> np.ndarray:
    """
    Encode texts in small batches to prevent memory spikes
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Encode batch
        with torch.no_grad():  # Disable gradient computation
            batch_embeddings = embedder.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        all_embeddings.append(batch_embeddings)
        
        # Clear cache
        if i % 32 == 0:
            gc.collect()
    
    return np.vstack(all_embeddings) if all_embeddings else np.array([])


# ── OPTIMISED ADD DOCUMENT ────────────────────────
def add_document(
    text: str,
    doc_id: str,
    filename: str,
    chunk_size: int = 50  # Reduced from 80 to 50
) -> Dict:
    """
    Memory-optimised document addition
    - Processes in smaller batches
    - Clears memory frequently
    - Smaller chunk size
    """
    steps = []
    total_start = time.time()

    # Step 1 — Chunk (memory efficient)
    t0 = time.time()
    chunks = chunk_text_streaming(text, chunk_size=chunk_size, overlap=10)
    
    steps.append({
        "step": "chunking",
        "label": "Text Chunking",
        "detail": f"{len(chunks)} chunks created ({chunk_size} words each)",
        "time_ms": round((time.time() - t0) * 1000, 1),
        "data": [{"chunk_no": c["chunk_no"], "words": c["words"],
                  "preview": c["text"][:50] + "..."} for c in chunks[:3]]
    })

    # Step 2 — Embed (batch processed)
    t0 = time.time()
    texts = [c["text"] for c in chunks]
    
    # Process embeddings in batches
    embeddings = batch_encode(texts, batch_size=8)
    
    embed_time = round((time.time() - t0) * 1000, 1)
    steps.append({
        "step": "embedding",
        "label": "Embedding Generation",
        "detail": f"{len(embeddings)} vectors · dim=384",
        "time_ms": embed_time,
        "data": {
            "model": EMBED_MODEL,
            "dimensions": 384,
            "count": len(embeddings)
        }
    })

    # Step 3 — Store in ChromaDB (batch upsert)
    t0 = time.time()
    col = get_collection(doc_id)
    
    # Prepare data
    ids = [f"{doc_id}_chunk_{c['chunk_no']}" for c in chunks]
    metadatas = [{
        "doc_id": doc_id,
        "filename": filename,
        "chunk_no": c["chunk_no"],
        "words": c["words"]
    } for c in chunks]
    
    # Upsert in batches to prevent memory spikes
    batch_size = 20
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        col.upsert(
            ids=ids[i:batch_end],
            documents=texts[i:batch_end],
            embeddings=embeddings[i:batch_end].tolist(),
            metadatas=metadatas[i:batch_end]
        )
        
        # Clear batch data
        if i % 40 == 0:
            gc.collect()

    steps.append({
        "step": "storing",
        "label": "ChromaDB Storage",
        "detail": f"Stored {len(chunks)} vectors in collection",
        "time_ms": round((time.time() - t0) * 1000, 1),
    })

    # Clear large variables
    del texts, embeddings, chunks
    gc.collect()

    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_words": len(text.split()),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "total_ms": round((time.time() - total_start) * 1000, 1),
        "steps": steps
    }


# ── OPTIMISED RETRIEVAL ───────────────────────────
def retrieve_and_rerank(
    query: str,
    doc_ids: List[str],
    top_k_embed: int = 10,  # Reduced from 20
    top_k_final: int = 3,    # Reduced from 5
    rerank_passes: int = 1    # Reduced from 2
) -> Dict:
    """
    Memory-optimised retrieval with fewer candidates
    """
    steps = []
    total_start = time.time()

    # Step 1 — Embed query
    t0 = time.time()
    with torch.no_grad():
        query_emb = embedder.encode(query).tolist()
    steps.append({
        "step": "query_embed",
        "label": "Query Embedding",
        "time_ms": round((time.time() - t0) * 1000, 1),
    })

    # Step 2 — Search (limited candidates)
    t0 = time.time()
    candidates = []
    
    for doc_id in doc_ids[:3]:  # Limit to 3 docs at a time
        try:
            col = get_collection(doc_id)
            if col.count() == 0:
                continue
            
            n = min(top_k_embed, col.count())
            results = col.query(
                query_embeddings=[query_emb],
                n_results=n
            )
            
            for i, doc in enumerate(results["documents"][0]):
                candidates.append({
                    "text": doc,
                    "doc_id": doc_id,
                    "chunk_no": results["metadatas"][0][i].get("chunk_no", i),
                    "filename": results["metadatas"][0][i].get("filename", doc_id),
                })
        except:
            continue

    steps.append({
        "step": "search",
        "label": "Vector Search",
        "detail": f"Found {len(candidates)} candidates",
        "time_ms": round((time.time() - t0) * 1000, 1),
    })

    if not candidates:
        return {"chunks": [], "steps": steps, "context": ""}

    # Step 3 — Single rerank pass
    t0 = time.time()
    
    # Limit candidates for reranking
    candidates = candidates[:top_k_embed]
    
    if rerank_passes >= 1 and candidates:
        pairs = [[query, c["text"][:500]] for c in candidates]  # Truncate text
        with torch.no_grad():
            scores = reranker.predict(pairs, batch_size=8)
        
        for i, c in enumerate(candidates):
            c["score"] = round(float(scores[i]), 4)
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        final_chunks = candidates[:top_k_final]
    else:
        final_chunks = candidates[:top_k_final]

    steps.append({
        "step": "rerank",
        "label": "Reranking",
        "time_ms": round((time.time() - t0) * 1000, 1),
    })

    # Step 4 — Build context
    context = "\n\n".join([c["text"] for c in final_chunks])

    # Clear memory
    gc.collect()

    return {
        "chunks": final_chunks,
        "context": context,
        "steps": steps,
        "total_ms": round((time.time() - total_start) * 1000, 1),
        "candidates": len(candidates),
        "final": len(final_chunks)
    }


def build_prompt(query: str, context: str) -> str:
    if not context:
        return query
    return f"""Context: {context[:1500]}

Question: {query}

Answer:"""


def get_all_docs() -> List[Dict]:
    return list_collections()


def get_doc_chunks(doc_id: str) -> List[Dict]:
    """Get chunks with pagination"""
    col = get_collection(doc_id)
    results = col.get(limit=50)  # Limit to 50 chunks
    chunks = []
    
    for i, doc in enumerate(results["documents"]):
        chunks.append({
            "chunk_no": i,
            "text": doc[:200],  # Truncate preview
            "preview": doc[:100]
        })
    
    return chunks


def get_stats() -> Dict:
    cols = chroma.list_collections()
    total_chunks = 0
    
    # Count chunks efficiently
    for c in cols[:10]:  # Limit to 10 collections for stats
        try:
            total_chunks += chroma.get_collection(c.name).count()
        except:
            continue
    
    return {
        "total_docs": len(cols),
        "total_chunks": total_chunks,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL
    }