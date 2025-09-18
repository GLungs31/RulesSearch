# athletics_rag_bot.py
# PyCharm-friendly RAG chatbot for state HS athletic handbooks/bylaws.
# - Put PDFs in ./handbooks
# - Run this file directly (green Run button) to start the API on http://127.0.0.1:8000
# - Open http://127.0.0.1:8000/docs to try /ask and /upload

from threading import Thread, Lock
from fastapi import HTTPException
import uuid, time

import os
import re
import io
import json
import uuid
import string
import hashlib
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import fitz  # PyMuPDF

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# --- Optional OCR (scanned PDFs) ---
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# --- Optional FAISS (faster vector search); falls back to NumPy if missing ---
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    faiss = None  # type: ignore
    FAISS_OK = False

# --- BM25 keyword search ---
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --------------------- Config ---------------------
DATA_DIR = os.getenv("HB_DATA_DIR", "./handbooks")
STORE_DIR = os.getenv("HB_STORE_DIR", "./store")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORE_DIR, exist_ok=True)

FAISS_PATH = os.path.join(STORE_DIR, "handbooks.faiss")
EMB_PATH   = os.path.join(STORE_DIR, "embeddings.npy")
META_PATH  = os.path.join(STORE_DIR, "metadata.json")

# --------------------- FastAPI app ---------------------
app = FastAPI(title="HS Athletics Handbook Bot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# (optional) serve PDFs so source links can open to exact page
app.mount("/handbooks", StaticFiles(directory=DATA_DIR), name="handbooks")

# --------------------- Background ingest plumbing ---------------------
INGEST_JOBS = {}
INGEST_LOCK = Lock()

def _ingest_worker(job_id: str):
    """Runs in background: reads PDFs and builds the index."""
    try:
        with INGEST_LOCK:
            INGEST_JOBS[job_id] = {**INGEST_JOBS.get(job_id, {}), "status": "running"}

        corpus = build_corpus(DATA_DIR)        # you already have this
        INDEX.build(corpus, persist=True)      # uses your HybridIndex

        files = len({c["metadata"]["file"] for c in corpus}) if corpus else 0
        with INGEST_LOCK:
            INGEST_JOBS[job_id].update(
                status="done",
                chunks=len(corpus),
                files=files,
                finished_at=time.time(),
                message="Index built."
            )
    except Exception as e:
        with INGEST_LOCK:
            INGEST_JOBS[job_id].update(
                status="error",
                error=str(e),
                finished_at=time.time()
            )

# --------------------- Endpoints ---------------------

# Health check
@app.get("/health")
def health():
    return {"ok": True, "docs": "/docs"}

# Save-only upload (fast) — do NOT index here
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files are accepted."}
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as out:
        out.write(file.file.read())
    return {"ok": True, "stored_as": dest, "message": "Uploaded. Now run POST /ingest-start."}
# ---- background ingest support ----
# tiny job store
INGEST_JOBS = {}          # job_id -> {status, chunks, files, ...}
INGEST_LOCK = Lock()

# worker that does the heavy lifting
def _ingest_worker(job_id: str):
    try:
        with INGEST_LOCK:
            INGEST_JOBS[job_id] = {"status": "running", "started_at": time.time()}
        corpus = build_corpus(DATA_DIR)        # uses your existing function
        INDEX.build(corpus, persist=True)      # uses your existing HybridIndex
        files = len({c["metadata"]["file"] for c in corpus}) if corpus else 0
        with INGEST_LOCK:
            INGEST_JOBS[job_id].update(
                status="done",
                chunks=len(corpus),
                files=files,
                finished_at=time.time(),
                message="Index built."
            )
    except Exception as e:
        with INGEST_LOCK:
            INGEST_JOBS[job_id] = {
                "status": "error",
                "error": str(e),
                "finished_at": time.time()
            }

# start ingest (returns immediately with a job_id)
@app.post("/ingest-start")
def ingest_start():
    job_id = str(uuid.uuid4())
    with INGEST_LOCK:
        INGEST_JOBS[job_id] = {"status": "queued", "started_at": time.time()}
    Thread(target=_ingest_worker, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "tip": "Poll /ingest-status/{job_id} until status=done."}

# check ingest status
@app.get("/ingest-status/{job_id}")
def ingest_status(job_id: str):
    job = INGEST_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

# Start ingest in background (returns job_id quickly)
@app.post("/ingest-start")
def ingest_start():
    job_id = str(uuid.uuid4())
    with INGEST_LOCK:
        INGEST_JOBS[job_id] = {"status": "queued", "started_at": time.time()}
    Thread(target=_ingest_worker, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "tip": "Poll /ingest-status/{job_id} until status=done."}

# Poll ingest status
@app.get("/ingest-status/{job_id}")
def ingest_status(job_id: str):
    job = INGEST_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/ingest-status/{job_id}")
def ingest_status(job_id: str):
    job = INGEST_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

# Save-only upload (fast) — use this to upload PDFs
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files are accepted."}
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as out:
        out.write(file.file.read())
    # IMPORTANT: do NOT index here. Keep upload fast to avoid 502s.
    return {"ok": True, "stored_as": dest, "message": "Uploaded. Now run /ingest-start."}

# Start ingest in background (returns a job_id immediately)
@app.post("/ingest-start")

# Poll ingest job status
@app.get("/ingest-status/{job_id}")
def ingest_status(job_id: str):
    job = INGEST_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

# Your existing /ask endpoint should already exist. If not, here’s a minimal one:
class AskRequest(BaseModel):
    question: str
    state: Optional[str] = None
    year: Optional[str] = None
    doc_title: Optional[str] = None
    top_k: int = 8

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # build filters
    filters = {}
    if req.state: filters["state"] = req.state
    if req.year: filters["year"] = req.year
    if req.doc_title: filters["doc_title"] = req.doc_title

    # run retrieval
    hits = INDEX.hybrid_search(req.question, k=max(3, min(req.top_k, 15)), filters=filters)

    # compose answer (use your existing compose_extractive_answer function)
    answer, cites = compose_extractive_answer(req.question, hits)
    return AskResponse(answer=answer, sources=cites)

# serve your PDFs so citations can link to pages (optional but nice)
app.mount("/handbooks", StaticFiles(directory=DATA_DIR), name="handbooks")

INGEST_JOBS: Dict[str, Dict[str, Any]] = {}
INGEST_LOCK = Lock()

def _ingest_worker(job_id: str):
    """Background job: read PDFs, build the index, and save it."""
    from collections import defaultdict  # safe to import here if you want

    try:
        with INGEST_LOCK:
            INGEST_JOBS[job_id] = {**INGEST_JOBS.get(job_id, {}), "status": "running"}

        # ---- build the corpus (you likely already have these functions) ----
        corpus = build_corpus(DATA_DIR)  # <-- your existing function
        INDEX.build(corpus, persist=True)  # <-- your existing HybridIndex.build

        files = len({c["metadata"]["file"] for c in corpus}) if corpus else 0
        with INGEST_LOCK:
            INGEST_JOBS[job_id].update(
                status="done",
                chunks=len(corpus),
                files=files,
                finished_at=time.time(),
                message="Index built."
            )
    except Exception as e:
        with INGEST_LOCK:
            INGEST_JOBS[job_id].update(
                status="error",
                error=str(e),
                finished_at=time.time()
            )

# --------------------- Helpers --------------------
# ========= SAFE, DROP-IN HYBRID INDEX (with empty-text guard) =========
# Requires: from sentence_transformers import SentenceTransformer
#           from rank_bm25 import BM25Okapi
#           try: import faiss; FAISS_OK=True except: FAISS_OK=False
# Also needs STORE_DIR to exist (we set default if missing)

# defaults if not set earlier
STORE_DIR = os.getenv("HB_STORE_DIR", "./store")
os.makedirs(STORE_DIR, exist_ok=True)
FAISS_PATH = os.path.join(STORE_DIR, "handbooks.faiss")
EMB_PATH   = os.path.join(STORE_DIR, "embeddings.npy")
META_PATH  = os.path.join(STORE_DIR, "metadata.json")
EMBED_MODEL_NAME = os.getenv("HB_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class HybridIndex:
    def __init__(self, embed_model_name: str = EMBED_MODEL_NAME):
        self.embed_model_name = embed_model_name
        self.model = None                # SentenceTransformer
        self.faiss_index = None          # FAISS index (optional)
        self.embeddings = None           # numpy array fallback
        self.corpus = []                 # list of {text, metadata}
        self.texts = []                  # list of strings
        self.tokens = []                 # list[list[str]] for BM25
        self.bm25 = None                 # BM25Okapi

    # --------------- BUILD ----------------
    def build(self, corpus: List[Dict[str, Any]], persist: bool = True) -> None:
        # Keep metadata and only non-empty texts
        self.corpus = corpus or []
        self.texts = [c.get("text", "") for c in self.corpus if (c.get("text", "").strip())]

        # Build BM25 even if small
        self.tokens = [self._tokenize_for_bm25(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens) if self.texts else None

        # ⛑️ Empty-text guard: do NOT try to embed if nothing to index
        if not self.texts:
            self.faiss_index = None
            self.embeddings = None
            self.model = None
            if persist:
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.corpus, f, ensure_ascii=False)
            print("No text to index (0 chunks). Upload PDFs and run /ingest-start later.")
            return

        # Build embeddings (only if we have text)
        self.model = SentenceTransformer(self.embed_model_name)
        embs = self.model.encode(self.texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")

        # Prefer FAISS; else keep numpy
        if 'FAISS_OK' in globals() and FAISS_OK:
            dim = embs.shape[1]
            import faiss  # safe: only if available
            index = faiss.IndexFlatIP(dim)
            index.add(embs)
            self.faiss_index = index
            if persist:
                faiss.write_index(self.faiss_index, FAISS_PATH)
                np.save(EMB_PATH, embs)
        else:
            self.embeddings = embs
            if persist:
                np.save(EMB_PATH, embs)

        if persist:
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(self.corpus, f, ensure_ascii=False)

    # --------------- LOAD -----------------
    def load(self) -> bool:
        # load metadata first
        if not os.path.exists(META_PATH):
            return False
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
            self.texts = [c.get("text", "") for c in self.corpus if (c.get("text", "").strip())]

            # BM25 (ok if empty)
            self.tokens = [self._tokenize_for_bm25(t) for t in self.texts]
            self.bm25 = BM25Okapi(self.tokens) if self.texts else None

            # Embedding model only needed for queries if we’ll use vectors
            self.model = SentenceTransformer(self.embed_model_name) if self.texts else None

            # Try FAISS first
            if 'FAISS_OK' in globals() and FAISS_OK and os.path.exists(FAISS_PATH):
                import faiss
                self.faiss_index = faiss.read_index(FAISS_PATH)
                self.embeddings = None
                return True

            # Else numpy embeddings
            if os.path.exists(EMB_PATH):
                self.embeddings = np.load(EMB_PATH)
                self.faiss_index = None
                return True

            # Nothing persisted beyond metadata
            return bool(self.texts)
        except Exception:
            return False

    # --------------- SEARCH ---------------
    def hybrid_search(
        self,
        query: str,
        k: int = 8,
        filters: Optional[Dict[str, str]] = None,
        bm25_weight: float = 0.45,
        vect_weight: float = 0.55
    ) -> List[Dict[str, Any]]:
        # If index not ready, return empty
        if not self.texts or (self.bm25 is None and self.faiss_index is None and self.embeddings is None):
            return []

        filters = filters or {}
        candidates = self._filter_candidates(filters)

        # Vector scores
        vec_pairs = []
        if self.model is not None and (self.faiss_index is not None or self.embeddings is not None):
            q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
            if self.faiss_index is not None:
                topN = min(k * 4, len(self.texts))
                scores, idxs = self.faiss_index.search(q_emb.reshape(1, -1), topN)
                vec_pairs = list(zip(idxs[0].tolist(), scores[0].tolist()))
            elif self.embeddings is not None:
                sims = self.embeddings @ q_emb
                idxs = np.argsort(-sims)[:min(k * 4, len(self.texts))]
                vec_pairs = [(int(i), float(sims[int(i)])) for i in idxs]

            if candidates is not None:
                vec_pairs = [(i, s) for i, s in vec_pairs if i in candidates]

        # Normalize vector scores to [0..1]
        vec_norm = [((s + 1.0) / 2.0) for _, s in vec_pairs] if vec_pairs else []

        # BM25 scores
        q_tokens = self._tokenize_for_bm25(query)
        if self.bm25 is None:
            bm_pairs = []
            bm_norm = {}
        else:
            bm_scores = self.bm25.get_scores(q_tokens)
            bm_pairs = list(enumerate(bm_scores))
            if candidates is not None:
                bm_pairs = [(i, sc) for i, sc in bm_pairs if i in candidates]
            if bm_pairs:
                vals = [sc for _, sc in bm_pairs]
                mn, mx = min(vals), max(vals)
                rng = (mx - mn) or 1.0
                bm_norm = {i: (sc - mn) / rng for i, sc in bm_pairs}
            else:
                bm_norm = {}

        # Combine
        combined: Dict[int, float] = {}
        for (i, s), ns in zip(vec_pairs, vec_norm):
            combined[i] = combined.get(i, 0.0) + vect_weight * ns
        for i, _ in bm_pairs:
            combined[i] = combined.get(i, 0.0) + bm25_weight * bm_norm.get(i, 0.0)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in ranked:
            hit = self.corpus[idx]
            results.append({"score": float(score), "text": hit["text"], "metadata": hit["metadata"]})
        return results

    # --------------- HELPERS --------------
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def _filter_candidates(self, filters: Dict[str, str]) -> Optional[set]:
        if not filters:
            return None
        cands = set()
        for i, c in enumerate(self.corpus):
            m = c.get("metadata", {})
            ok = True
            for k, v in filters.items():
                if not v:
                    continue
                have = (m.get(k, "") or "")
                if str(v).lower() not in str(have).lower():
                    ok = False
                    break
            if ok:
                cands.add(i)
        # if nothing matched, fall back to unfiltered
        return cands if cands else None

# Make a global index instance you can use everywhere
INDEX = HybridIndex()
# ========= END HYBRID INDEX =========

HEAD_RE = re.compile(r"^(Article|Rule|Section|Bylaw|Policy)\s+[\w\.\-]+.*", re.I | re.M)

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def guess_metadata_from_filename(fname: str) -> Dict[str, str]:
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split("_")
    state = parts[0] if parts else ""
    year_match = re.search(r"(20\d{2}(?:-\d{2})?)", stem)
    year = year_match.group(1) if year_match else ""
    return {"state": state, "year": year, "doc_title": stem, "file": os.path.basename(fname)}

# --------------------- PDF → chunks -------------------
def extract_page_text(page: fitz.Page) -> str:
    text = page.get_text("text") or ""
    if text.strip():
        return text
    if not OCR_AVAILABLE:
        return text
    try:
        pix = page.get_pixmap(dpi=300, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return text

def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        txt = extract_page_text(page)
        pages.append({"page": i + 1, "page_number": i + 1, "text": txt})  # << fix here
    doc.close()
    return pages

def chunk_text(text: str, page_number: int, max_chars=1400, overlap=200) -> List[Dict[str, Any]]:
    chunks = []
    text = normalize_space(text)
    if not text:
        return chunks
    heads = list(HEAD_RE.finditer(text))
    if heads:
        boundaries = [m.start() for m in heads] + [len(text)]
        for s, e in zip(boundaries, boundaries[1:]):
            t = text[s:e].strip()
            if t:
                chunks.append({"text": t, "page": page_number})
    else:
        i = 0
        while i < len(text):
            t = text[i:i + max_chars].strip()
            if t:
                chunks.append({"text": t, "page": page_number})
            i += max_chars - overlap
    return chunks

def build_corpus(data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(data_dir, fname)
        meta = guess_metadata_from_filename(fname)
        meta["checksum"] = file_checksum(fpath)
        pages = extract_pages(fpath)
        for p in pages:
            for c in chunk_text(p["text"], p["page"]):
                corpus.append({
                    "id": str(uuid.uuid4()),
                    "text": c["text"],
                    "metadata": {**meta, "page": c["page"]},
                })
    return corpus

# --------------------- Hybrid Index --------------------
class HybridIndex:
    def __init__(self, embed_model_name: str = EMBED_MODEL_NAME):
        self.embed_model_name = embed_model_name
        self.model: Optional[SentenceTransformer] = None
        self.faiss_index = None
        self.embeddings: Optional[np.ndarray] = None  # fallback if FAISS not present
        self.corpus: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build(self, corpus: List[Dict[str, Any]], persist: bool = True) -> None:
        self.corpus = corpus or []
        self.texts = [c.get("text", "") for c in self.corpus if (c.get("text", "").strip())]
        self.tokens = [self._tokenize_for_bm25(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens) if self.texts else None

        if not self.texts:
            self.faiss_index = None
            self.embeddings = None
            self.model = None
            if persist:
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.corpus, f, ensure_ascii=False)
            print("No text to index (0 chunks). Upload PDFs and call /ingest later.")
            return

        self.corpus = corpus
        self.texts = [c["text"] for c in corpus]

        # Embeddings
        self.model = SentenceTransformer(self.embed_model_name)
        embs = self.model.encode(self.texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")
        if hasattr(embs, 'shape') and len(embs.shape) > 1:
            dim = embs.shape[1]
        else:
            print("ERROR: Expected embeddings to be 2D, got shape", getattr(embs, 'shape', None))
            # You might want to raise an exception or skip building the index
            return
        if FAISS_OK:
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors (dot product)
            index.add(embs)
            self.faiss_index = index
            if persist:
                faiss.write_index(self.faiss_index, FAISS_PATH)
                np.save(EMB_PATH, embs)  # also keep a copy for safety
        else:
            self.embeddings = embs
            if persist:
                np.save(EMB_PATH, embs)

        if persist:
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(self.corpus, f, ensure_ascii=False)

        # BM25
        self.tokens = [self._tokenize_for_bm25(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens)

    def load(self) -> bool:
        if not os.path.exists(META_PATH):
            return False
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
            self.texts = [c["text"] for c in self.corpus]
            self.model = SentenceTransformer(self.embed_model_name)

            if FAISS_OK and os.path.exists(FAISS_PATH):
                self.faiss_index = faiss.read_index(FAISS_PATH)
            elif os.path.exists(EMB_PATH):
                self.embeddings = np.load(EMB_PATH)
            else:
                return False

            self.tokens = [self._tokenize_for_bm25(t) for t in self.texts]
            self.bm25 = BM25Okapi(self.tokens)
            return True
        except Exception:
            return False

    def hybrid_search(
        self,
        query: str,
        k: int = 8,
        filters: Optional[Dict[str, str]] = None,
        bm25_weight: float = 0.45,
        vect_weight: float = 0.55
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        candidates = self._filter_candidates(filters)

        # Vector search
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
        if self.faiss_index is not None:
            topN = min(k * 4, len(self.texts))
            scores, idxs = self.faiss_index.search(q_emb.reshape(1, -1), topN)
            vec_pairs = list(zip(idxs[0].tolist(), scores[0].tolist()))
        else:
            # NumPy dot product fallback
            if self.embeddings is None or len(self.texts) == 0:
                vec_pairs = []
            else:
                sims = self.embeddings @ q_emb
                idxs = np.argsort(-sims)[:min(k * 4, len(self.texts))]
                vec_pairs = [(int(i), float(sims[int(i)])) for i in idxs]

        if candidates is not None:
            vec_pairs = [(i, s) for i, s in vec_pairs if i in candidates]
        vec_norm = [((s + 1.0) / 2.0) for _, s in vec_pairs] if vec_pairs else []

        # BM25
        q_tokens = self._tokenize_for_bm25(query)
        bm_scores = self.bm25.get_scores(q_tokens)
        bm_pairs = list(enumerate(bm_scores))
        if candidates is not None:
            bm_pairs = [(i, sc) for i, sc in bm_pairs if i in candidates]
        if bm_pairs:
            vals = [sc for _, sc in bm_pairs]
            mn, mx = min(vals), max(vals)
            rng = (mx - mn) or 1.0
            bm_norm = {i: (sc - mn) / rng for i, sc in bm_pairs}
        else:
            bm_norm = {}

        # Combine
        combined: Dict[int, float] = {}
        for (i, s), ns in zip(vec_pairs, vec_norm):
            combined[i] = combined.get(i, 0.0) + vect_weight * ns
        for i, _ in bm_pairs:
            combined[i] = combined.get(i, 0.0) + bm25_weight * bm_norm.get(i, 0.0)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in ranked:
            hit = self.corpus[idx]
            results.append({"score": float(score), "text": hit["text"], "metadata": hit["metadata"]})
        return results

    # ---- helpers ----
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def _filter_candidates(self, filters: Dict[str, str]) -> Optional[set]:
        if not filters:
            return None
        cands = set()
        for i, c in enumerate(self.corpus):
            m = c["metadata"]
            ok = True
            for k, v in filters.items():
                if not v:
                    continue
                have = (m.get(k, "") or "")
                if str(v).lower() not in str(have).lower():
                    ok = False
                    break
            if ok:
                cands.add(i)
        # 👇 If no candidates match, return None so we still get unfiltered results
        return cands if cands else None

    def _filter_candidates(self, filters: Dict[str, str]) -> Optional[set]:
        if not filters:
            return None
        cands = set()
        for i, c in enumerate(self.corpus):
            m = c["metadata"]
            ok = True
            for k, v in filters.items():
                if not v:
                    continue
                have = (m.get(k, "") or "")
                if str(v).lower() not in str(have).lower():
                    ok = False
                    break
            if ok:
                cands.add(i)
        return cands

# --------------------- Answer composer ------------------
def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]

def score_sentence(sent: str, q_terms: List[str]) -> float:
    s = sent.lower()
    return sum(1.0 for t in q_terms if t in s) / (len(q_terms) or 1)

def compose_extractive_answer(query: str, passages: List[Dict[str, Any]], max_sentences:int=6):
    # Try to pick sentences that overlap with the query terms
    q_tokens = [w for w in re.sub(r"[^\w\s]", " ", query.lower()).split() if len(w) > 2]
    pool = []
    for p in passages[:6]:
        m = p["metadata"]
        for s in split_sentences(p["text"]):
            sc = score_sentence(s, q_tokens)
            if sc > 0:
                pool.append((sc, s, m))
    pool.sort(key=lambda x: x[0], reverse=True)

    bullets, cites = [], []

    # Happy path: we found overlapping sentences
    if pool:
        seen = set()
        for _, s, m in pool:
            key = s[:140]
            if key in seen:
                continue
            seen.add(key)
            cite = f"({m.get('state','')}, {m.get('doc_title','')}, p. {m.get('page','')})"
            bullets.append(f"- {s} {cite}")
            cites.append({
                "state": m.get("state",""),
                "doc_title": m.get("doc_title",""),
                "page": m.get("page",""),
                "file": m.get("file",""),
                "snippet": s,
            })
            if len(bullets) >= max_sentences:
                break
        answer = "Here’s what the indexed handbooks/bylaws say (with page citations):\n" + "\n".join(bullets)
        return answer, cites

    # Fallback: show top retrieved snippets even if the word overlap is weak
    if passages:
        for p in passages[:3]:
            m = p["metadata"]
            snippet = (p["text"][:300] + "…") if len(p["text"]) > 300 else p["text"]
            cite = f"({m.get('state','')}, {m.get('doc_title','')}, p. {m.get('page','')})"
            bullets.append(f"- {snippet} {cite}")
            cites.append({
                "state": m.get("state",""),
                "doc_title": m.get("doc_title",""),
                "page": m.get("page",""),
                "file": m.get("file",""),
                "snippet": snippet,
            })
        answer = "Top relevant excerpts (showing best matches we found):\n" + "\n".join(bullets)
        return answer, cites

    # No passages at all
    return ("I couldn’t find any content for that query. Try adding a state (e.g., “CT”) or a term like “transfer”, “age”, “semester”, or “bona fide residence.”", [])


# --------------------- API schemas ----------------------
class AskRequest(BaseModel):
    question: str = Field(..., example="What is the transfer eligibility rule after changing schools?")
    state: Optional[str] = Field(None, example="CT")
    year: Optional[str] = Field(None, example="2024-25")
    doc_title: Optional[str] = Field(None, example="CIAC_Handbook_2024-25")
    top_k: int = 8

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class IngestResponse(BaseModel):
    chunks_indexed: int
    files_seen: int

# --------------------- FastAPI app ----------------------
app = FastAPI(title="Athletics Handbooks Chatbot", version="1.0.0")
def _ingest_worker(job_id: str):
    try:
        with INGEST_LOCK:
            INGEST_JOBS[job_id] = {**INGEST_JOBS.get(job_id, {}), "status": "running"}

        # build the index (these two functions already exist in your file)
        corpus = build_corpus(DATA_DIR)
        INDEX.build(corpus, persist=True)

        files = len({c["metadata"]["file"] for c in corpus}) if corpus else 0
        with INGEST_LOCK:
            INGEST_JOBS[job_id].update(
                status="done",
                chunks=len(corpus),
                files=files,
                finished_at=time.time(),
                message="Index built."
            )
    except Exception as e:
        with INGEST_LOCK:
            INGEST_JOBS[job_id].update(
                status="error",
                error=str(e),
                finished_at=time.time()
            )

INGEST_JOBS = {}
INGEST_LOCK = Lock()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

INDEX = HybridIndex()

def ensure_index_loaded() -> None:
    if INDEX.load():
        print("Index loaded.")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}. Upload via /upload then run /ingest-start.")
        return
    corpus = build_corpus(DATA_DIR)
    INDEX.build(corpus, persist=True)
    print(f"Built index with {len(corpus)} chunks.")
@app.on_event("startup")
def _startup():
    ensure_index_loaded()   # must SKIP building if no PDFs exist

@app.get("/health")
def health():
    return {"ok": True, "docs": "/docs"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    filters = {}
    if req.state: filters["state"] = req.state
    if req.year: filters["year"] = req.year
    if req.doc_title: filters["doc_title"] = req.doc_title
    hits = INDEX.hybrid_search(req.question, k=max(3, min(req.top_k, 15)), filters=filters)
    answer, cites = compose_extractive_answer(req.question, hits)
    return AskResponse(answer=answer, sources=cites)

@app.post("/ingest", response_model=IngestResponse)
def ingest_all():
    print("Ingest started…")
    corpus = build_corpus(DATA_DIR)
    INDEX.build(corpus, persist=True)
    print("Ingest finished.")
    return IngestResponse(
        chunks_indexed=len(corpus),
        files_seen=len({c["metadata"]["file"] for c in corpus})
    )

@app.post("/ingest", response_model=IngestResponse)
def ingest_all():
    corpus = build_corpus(DATA_DIR)
    INDEX.build(corpus, persist=True)
    return IngestResponse(chunks_indexed=len(corpus), files_seen=len({c["metadata"]["file"] for c in corpus}))

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files are accepted."}
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as out:
        out.write(file.file.read())
    # do NOT index here—keeps upload fast and avoids 502s on Render
    return {"ok": True, "stored_as": dest, "message": "Uploaded. Now run POST /ingest-start."}

# --------------------- Run with PyCharm -----------------
# --- DEBUG: corpus stats and search preview ---
from collections import defaultdict
from fastapi import Body

@app.get("/corpus-stats")
def corpus_stats():
    by_state = defaultdict(int)
    by_file = defaultdict(lambda: {"doc_title":"", "year":"", "state":"", "chunks":0, "max_page":0})
    for c in INDEX.corpus:
        m = c.get("metadata", {})
        st = (m.get("state") or "").strip()
        by_state[st] += 1
        f = m.get("file") or ""
        row = by_file[f]
        row["doc_title"] = m.get("doc_title") or row["doc_title"]
        row["year"] = m.get("year") or row["year"]
        row["state"] = st or row["state"]
        row["chunks"] += 1
        try:
            row["max_page"] = max(int(m.get("page") or 0), row["max_page"])
        except Exception:
            pass
    return {
        "states": by_state,
        "docs": [{"file":k, **v} for k, v in by_file.items()]
    }

class DebugReq(BaseModel):
    query: str
    state: Optional[str] = None
    year: Optional[str] = None
    doc_title: Optional[str] = None
    k: int = 5

@app.post("/search-debug")
def search_debug(req: DebugReq):
    filters = {}
    if req.state: filters["state"] = req.state
    if req.year: filters["year"] = req.year
    if req.doc_title: filters["doc_title"] = req.doc_title
    hits = INDEX.hybrid_search(req.query, k=req.k, filters=filters)
    out = []
    for h in hits:
        m = h["metadata"]
        out.append({
            "score": round(h["score"], 3),
            "state": m.get("state",""),
            "year": m.get("year",""),
            "doc_title": m.get("doc_title",""),
            "page": m.get("page",""),
            "file": m.get("file",""),
            "preview": (h["text"][:400] + "…") if len(h["text"])>400 else h["text"]
        })
    return {"results": out}
@app.post("/ask-simple")
def ask_simple(req: AskRequest):
    filters = {}
    if req.state: filters["state"] = req.state
    if req.year: filters["year"] = req.year
    if req.doc_title: filters["doc_title"] = req.doc_title
    hits = INDEX.hybrid_search(req.question, k=max(3, min(req.top_k, 15)), filters=filters)
    out = []
    for h in hits:
        m = h["metadata"]
        out.append({
            "score": round(h["score"], 3),
            "state": m.get("state",""),
            "year": m.get("year",""),
            "doc_title": m.get("doc_title",""),
            "page": m.get("page",""),
            "file": m.get("file",""),
            "preview": (h["text"][:400] + "…") if len(h["text"])>400 else h["text"]
        })
    return {"hits": out}
@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>HS Athletics Handbook Chat</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:0; background:#f5f6f8;}
  .wrap { max-width:900px; margin:0 auto; padding:20px; }
  .header { display:flex; gap:10px; align-items:center; margin-bottom:10px;}
  .chat { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px; height:60vh; overflow:auto;}
  .msg { margin:10px 0; display:flex; }
  .msg.user { justify-content:flex-end; }
  .bubble { padding:10px 12px; border-radius:12px; max-width:75%; white-space:pre-wrap; }
  .msg.user .bubble { background:#2563eb; color:white; border-bottom-right-radius:4px; }
  .msg.bot .bubble { background:#f1f5f9; color:#111827; border-bottom-left-radius:4px; }
  .sources { font-size:12px; margin-top:6px; }
  .sources a { margin-right:8px; }
  .composer { display:flex; gap:10px; margin-top:12px; }
  .composer input { flex:1; padding:10px; border-radius:10px; border:1px solid #cbd5e1;}
  .composer button { padding:10px 14px; border:none; background:#2563eb; color:#fff; border-radius:10px; cursor:pointer;}
  .small { font-size:12px; color:#64748b; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h2 style="margin:0;">HS Athletics Handbook Chat</h2>
    <span class="small">Try: "transfer after changing schools", "age eligibility", "coach ejection penalty"</span>
  </div>

  <div style="display:flex; gap:10px; margin: 8px 0 12px;">
    <input id="state" placeholder="State (e.g., CT)" style="width:120px;">
    <input id="year" placeholder="Year (e.g., 2024-25)" style="width:160px;">
    <input id="topk" type="number" min="3" max="15" value="8" style="width:90px;" title="top_k">
  </div>

  <div id="chat" class="chat"></div>

  <div class="composer">
    <input id="q" placeholder="Type your question and press Enter…" autocomplete="off">
    <button id="send">Send</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const q = document.getElementById('q');
const send = document.getElementById('send');

function addMsg(text, who='bot', sources=null) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.textContent = text;
  div.appendChild(bub);
  chat.appendChild(div);
  if (sources && sources.length) {
    const sdiv = document.createElement('div');
    sdiv.className = 'sources';
    sources.forEach((s, i) => {
      const file = s.file || '';
      const page = s.page || 1;
      const href = file ? `/handbooks/${encodeURIComponent(file)}#page=${page}` : (s.url || '#');
      const a = document.createElement('a');
      a.href = href; a.target = '_blank';
      a.textContent = `[${i+1}] ${s.state || ''} p.${page}`;
      sdiv.appendChild(a);
    });
    div.appendChild(sdiv);
  }
  chat.scrollTop = chat.scrollHeight;
}

async function ask() {
  const text = q.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  q.value = '';
  addMsg('Thinking…');

  const payload = {
    question: text,
    top_k: parseInt(document.getElementById('topk').value || '8', 10)
  };
  const st = document.getElementById('state').value.trim();
  const yr = document.getElementById('year').value.trim();
  if (st) payload.state = st;
  if (yr) payload.year = yr;

  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    chat.removeChild(chat.lastChild); // remove "Thinking…"
    addMsg(data.answer || '(no answer)', 'bot', data.sources || []);
  } catch (e) {
    chat.removeChild(chat.lastChild);
    addMsg('Error talking to the server. Is it still running?', 'bot');
  }
}

send.onclick = ask;
q.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
</script>
</body>
</html>
"""
if __name__ == "__main__":
    # Start the API directly so you can click Run in PyCharm
    import uvicorn
    print("Starting server at http://127.0.0.1:8000 (Docs: /docs)")
    if __name__ == "__main__":
        import uvicorn
        print("Starting server at http://127.0.0.1:8000 (Docs: /docs)")
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

