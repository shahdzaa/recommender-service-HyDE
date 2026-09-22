from __future__ import annotations
import json
import math
import re
import string
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5TokenizerFast

MODEL_DIR = Path(__file__).parent / "model"
GENERATOR_MODEL = "t5-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 64
CHUNK_SIZE = 40
TOP_CHUNKS_AVG = 3
NUM_BEAMS = 5
NUM_CANDIDATES = 5
TOP_N = 5

PROMPT_PREFIX = "Generate course title for this content:"

W_TITLE = 0.20
W_MATCHER = 0.75
W_GEN = 0.05


def clean(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    v = str(text).strip()
    return "" if v.lower() == "nan" else re.sub(r"\s+", " ", v).strip()


def normalize(text: Any) -> str:
    v = clean(text).lower()
    v = v.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", v).strip()


def slugify(value: str) -> str:
    value = clean(value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "course"


def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    words = clean(text).split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


class AppState:
    model: PeftModel
    tokenizer: T5TokenizerFast
    embedder: SentenceTransformer
    catalog: pd.DataFrame
    title_embs: torch.Tensor
    chunk_embs: torch.Tensor
    chunk_owners: torch.Tensor
    doc_boundaries: List[Tuple[int, int]]
    titles: List[str]
    title_to_idx: dict


state = AppState()


def load_best_weights() -> Tuple[float, float, float]:
    wt_path = MODEL_DIR / "weight_tuning.json"
    if wt_path.exists():
        data = json.loads(wt_path.read_text())
        w = data.get("best", {}).get("weights", [W_TITLE, W_MATCHER, W_GEN])
        return float(w[0]), float(w[1]), float(w[2])
    return W_TITLE, W_MATCHER, W_GEN


def build_index():
    print("⚡ Encoding catalog titles...")
    state.title_embs = state.embedder.encode(
        state.titles, convert_to_tensor=True, show_progress_bar=True, device=DEVICE
    )

    print("⚡ Chunking & encoding catalog documents...")
    docs = state.catalog["catalog_doc"].tolist()
    all_chunks, owners = [], []
    for i, doc in enumerate(docs):
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
        owners.extend([i] * len(chunks))

    state.chunk_owners = torch.tensor(owners, dtype=torch.long, device=DEVICE)
    state.chunk_embs = state.embedder.encode(
        all_chunks, convert_to_tensor=True, show_progress_bar=True,
        device=DEVICE, batch_size=64
    )

    boundaries = []
    cur = 0
    for doc_id in range(len(docs)):
        cnt = int((state.chunk_owners == doc_id).sum().item())
        boundaries.append((cur, cur + cnt))
        cur += cnt
    state.doc_boundaries = boundaries
    print(f"✅ Index ready — {len(state.titles)} courses, {len(all_chunks)} chunks")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading models & catalog...")
    global state

    wt, wm, wg = load_best_weights()
    state.W_TITLE = wt
    state.W_MATCHER = wm
    state.W_GEN = wg
    print(f"✅ Weights: title={wt}, matcher={wm}, gen_content={wg}")

    state.tokenizer = T5TokenizerFast.from_pretrained(str(MODEL_DIR))
    base = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL)
    state.model = PeftModel.from_pretrained(base, str(MODEL_DIR), is_trainable=False).to(DEVICE)
    state.model.eval()
    print("✅ T5 + LoRA loaded")

    state.embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    print("✅ Sentence Transformer loaded")

    state.catalog = pd.read_csv(MODEL_DIR / "catalog.csv")
    state.titles = state.catalog["course_title"].tolist()
    state.title_to_idx = {normalize(t): i for i, t in enumerate(state.titles)}
    print(f"✅ Catalog loaded — {len(state.titles)} courses")

    build_index()
    print("🎉 Service ready!")

    yield

    print("🛑 Shutting down...")


app = FastAPI(title="Masar Recommender", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    syllabus_text: str


class CourseRecommendation(BaseModel):
    rank: int
    course_title: str
    course_id: int
    course_slug: str
    score: float


class RecommendResponse(BaseModel):
    recommendations: List[CourseRecommendation]


@torch.no_grad()
def encode(texts: List[str]) -> torch.Tensor:
    return state.embedder.encode(
        texts, convert_to_tensor=True, show_progress_bar=False, device=DEVICE
    )


@torch.no_grad()
def chunk_similarity(query_embs: torch.Tensor) -> torch.Tensor:
    chunk_scores = util.cos_sim(query_embs, state.chunk_embs)
    bs = chunk_scores.size(0)
    num_docs = len(state.titles)
    doc_scores = torch.full((bs, num_docs), -1e9, device=DEVICE)
    for doc_id, (start, end) in enumerate(state.doc_boundaries):
        if start == end:
            continue
        top_k = min(TOP_CHUNKS_AVG, end - start)
        top_vals, _ = torch.topk(chunk_scores[:, start:end], top_k, dim=1)
        doc_scores[:, doc_id] = top_vals.mean(dim=1)
    return doc_scores


@torch.no_grad()
def recommend(syllabus_text: str, top_n: int = TOP_N) -> List[CourseRecommendation]:
    prompt = f"{PROMPT_PREFIX} {clean(syllabus_text)}"
    enc = state.tokenizer(
        prompt,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    gen_ids = state.model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=MAX_TARGET_LEN,
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_CANDIDATES,
        do_sample=False,
        early_stopping=True,
    )

    gen_titles = [
        state.tokenizer.decode(g, skip_special_tokens=True).strip() or "general course"
        for g in gen_ids
    ]

    all_scores = []
    for gt in gen_titles:
        gen_emb = encode([gt])
        mat_emb = encode([syllabus_text])

        title_sim = util.cos_sim(gen_emb, state.title_embs)
        matcher_sim = chunk_similarity(mat_emb)
        gen_sim = chunk_similarity(gen_emb)

        score = (
            state.W_TITLE * title_sim +
            state.W_MATCHER * matcher_sim +
            state.W_GEN * gen_sim
        )
        all_scores.append(score)

    combined = torch.stack(all_scores, dim=0).max(dim=0).values
    combined = combined.squeeze(0)

    top_scores, top_idx = torch.topk(combined, min(top_n, len(state.titles)))

    results = []
    for rank, (idx, score) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), start=1):
        course_row = state.catalog.iloc[idx]
        course_id = int(course_row["course_id"])
        course_slug = str(
            course_row.get("course_slug")
            or course_row.get("slug")
            or slugify(str(course_row["course_title"]))
        )

        results.append(CourseRecommendation(
            rank=rank,
            course_title=state.titles[idx],
            course_id=course_id,
            course_slug=course_slug,
            score=round(float(score), 4),
        ))
    return results

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "courses": len(state.titles)}


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend_endpoint(body: RecommendRequest):
    if not body.syllabus_text.strip():
        raise HTTPException(status_code=400, detail="syllabus_text is empty")
    try:
        recs = recommend(body.syllabus_text)
        return RecommendResponse(recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))