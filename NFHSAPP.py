# athletics_rag_bot.py
# PyCharm-friendly RAG chatbot for state HS athletic handbooks/bylaws.
# - Put PDFs in ./handbooks
# - Run this file directly (green Run button) to start the API on http://127.0.0.1:8000
# - Open http://127.0.0.1:8000/docs to try /ask and /upload

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
DATA_DIR = "./handbooks"     # put PDFs here
STORE_DIR = "./store"        # index/metadata here
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORE_DIR, exist_ok=True)

FAISS_PATH = os.path.join(STORE_DIR, "handbooks.faiss")
EMB_PATH   = os.path.join(STORE_DIR, "embeddings.npy")
META_PATH  = os.path.join(STORE_DIR, "metadata.json")

# --------------------- Helpers --------------------
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
@app.on_event("startup")
def _startup():
    ensure_index_loaded()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

INDEX = HybridIndex()

def ensure_index_loaded() -> None:
    # Try loading an existing index (OK if none yet)
    loaded = INDEX.load()
    if loaded:
        print("Index loaded.")
        return

    # If there are no PDFs yet, skip building (keep API alive)
    os.makedirs(DATA_DIR, exist_ok=True)
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}. Upload via /upload then run /ingest.")
        return

    # Build only if PDFs are present
    corpus = build_corpus(DATA_DIR)
    INDEX.build(corpus, persist=True)
    print(f"Built index with {len(corpus)} chunks from {len({c['metadata']['file'] for c in corpus})} files.")


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
    corpus = build_corpus(DATA_DIR)
    INDEX.build(corpus, persist=True)
    return IngestResponse(chunks_indexed=len(corpus), files_seen=len({c["metadata"]["file"] for c in corpus}))

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files are accepted."}
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as out:
        out.write(file.file.read())
    corpus = build_corpus(DATA_DIR)
    INDEX.build(corpus, persist=True)
    return {"ok": True, "stored_as": dest}

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

