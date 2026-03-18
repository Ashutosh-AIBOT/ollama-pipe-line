import os, asyncio, tempfile, time, httpx
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from rag import (add_document, retrieve_and_rerank, build_prompt,
                 get_all_docs, get_doc_chunks, get_stats)

# ── CONFIG ────────────────────────────────────────
API_KEY     = os.getenv("API_KEY",     "mypassword123")
OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://localhost:11434")
PIPER_VOICE = os.getenv("PIPER_VOICE", "./voices/en_US-lessac-medium.onnx")

# ── SINGLE MODEL FOR EVERYTHING ───────────────────
# phi3:mini is great for both chat and reasoning, only 2.2GB
MODEL = os.getenv("MODEL", "phi3:mini")

# ── ONE MODEL IN RAM AT A TIME ────────────────────
_lock         = asyncio.Lock()
_loaded_model = None


async def run_model(prompt: str) -> str:
    """
    Run the single model - no switching needed
    """
    global _loaded_model

    async with _lock:
        async with httpx.AsyncClient(timeout=180) as c:
            # run model
            try:
                r = await c.post(f"{OLLAMA_URL}/api/generate",
                    json={
                        "model"     : MODEL,
                        "prompt"    : prompt,
                        "stream"    : False,
                        "keep_alive": "5m",  # keeps in RAM for 5min
                        "options"   : {"num_ctx": 2048}
                    })
                r.raise_for_status()
                _loaded_model = MODEL
                print(f"Running {MODEL}")
                return r.json()["response"]
            except Exception as e:
                raise HTTPException(503, f"Model error: {e}")


# ── APP ───────────────────────────────────────────
app = FastAPI(title="RAG AI", version="3.1")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sec = APIKeyHeader(name="X-API-Key")
def verify(key: str = Depends(sec)):
    if key != API_KEY:
        raise HTTPException(401, "Wrong API Key")
    return key


# ── REQUEST MODELS ────────────────────────────────
class ChatReq(BaseModel):
    message      : str
    use_rag      : bool      = True
    doc_ids      : List[str] = []
    rerank_passes: int       = 2
    top_k        : int       = 5

class TTSReq(BaseModel):
    text: str


# ── HEALTH ────────────────────────────────────────
@app.get("/health")
async def health():
    result = {
        "api"          : "ok",
        "ollama"       : "error",
        "models"       : [],
        "loaded_model" : _loaded_model,
        "model"        : MODEL
    }
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            result["ollama"] = "ok"
            result["models"] = [m["name"] for m in r.json().get("models", [])]
    except:
        pass
    return result


# ── CHAT + RAG ────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatReq, _=Depends(verify)):
    total_start = time.time()

    # RAG
    rag_result = None
    context    = ""
    rag_steps  = []

    if req.use_rag and req.doc_ids:
        rag_result = retrieve_and_rerank(
            query        = req.message,
            doc_ids      = req.doc_ids,
            top_k_embed  = 20,
            top_k_final  = req.top_k,
            rerank_passes= req.rerank_passes
        )
        context   = rag_result.get("context", "")
        rag_steps = rag_result.get("steps", [])

    prompt    = build_prompt(req.message, context)
    llm_start = time.time()
    reply     = await run_model(prompt)
    llm_ms    = round((time.time() - llm_start) * 1000, 1)

    rag_steps.append({
        "step"   : "llm",
        "label"  : f"LLM — {MODEL}",
        "detail" : f"Single model for all tasks · auto-unloads after 5min idle",
        "time_ms": llm_ms,
        "data"   : {"model": MODEL, "prompt_len": len(prompt)}
    })

    return {
        "reply"       : reply,
        "model"       : MODEL,
        "rag_used"    : req.use_rag and bool(req.doc_ids),
        "chunks_found": rag_result["final"] if rag_result else 0,
        "candidates"  : rag_result["candidates"] if rag_result else 0,
        "rag_steps"   : rag_steps,
        "llm_ms"      : llm_ms,
        "total_ms"    : round((time.time() - total_start) * 1000, 1)
    }


# ── UPLOAD ────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...), chunk_size: int = 80, _: str = Depends(verify)):
    content = await file.read()
    fname   = file.filename
    if fname.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    elif fname.endswith(".pdf"):
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:
            raise HTTPException(400, f"PDF error: {e}")
    else:
        raise HTTPException(400, "Only .txt and .pdf supported")
    doc_id = fname.replace(" ", "_").replace(".", "_")
    result = add_document(text, doc_id, fname, chunk_size=chunk_size)
    return {"ok": True, **result}


# ── DOCS ──────────────────────────────────────────
@app.get("/docs")
async def list_docs(_=Depends(verify)):
    return {"docs": get_all_docs()}

@app.get("/docs/{doc_id}/chunks")
async def doc_chunks(doc_id: str, _=Depends(verify)):
    chunks = get_doc_chunks(doc_id)
    return {"doc_id": doc_id, "chunks": chunks, "total": len(chunks)}

@app.get("/stats")
async def stats(_=Depends(verify)):
    return {**get_stats(), "loaded_model": _loaded_model}


# ── TTS — independent, never blocks LLM ──────────
@app.post("/tts")
async def tts(req: TTSReq, _=Depends(verify)):
    tmp = tempfile.mktemp(suffix=".wav", dir="/tmp")
    try:
        proc = await asyncio.create_subprocess_shell(
            f'echo "{req.text}" | piper --model {PIPER_VOICE} --output_file {tmp}',
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.communicate()
        if not os.path.exists(tmp):
            raise HTTPException(500, "TTS failed")
        def stream():
            with open(tmp, "rb") as f:
                yield from f
            os.unlink(tmp)
        return StreamingResponse(stream(), media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=reply.wav"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── FRONTEND ──────────────────────────────────────
# Serve frontend with debug info
import os
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
print(f"\n{'='*50}")
print(f"🔍 Serving frontend from: {frontend_path}")
print(f"📁 Frontend exists: {os.path.exists(frontend_path)}")
if os.path.exists(frontend_path):
    print(f"📄 index.html exists: {os.path.exists(os.path.join(frontend_path, 'index.html'))}")
print(f"{'='*50}\n")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="ui")
else:
    print(f"❌ ERROR: Frontend path {frontend_path} does not exist!")
    print(f"📌 Current working directory: {os.getcwd()}")