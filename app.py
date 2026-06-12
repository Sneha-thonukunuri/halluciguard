"""
HalluciGuard — FastAPI backend
Runs GPT-2 generation + SelfCheckGPT hallucination scoring (ALL methods)
"""

import torch
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
from transformers import pipeline

from selfcheckgpt.modeling_selfcheck import (
    SelfCheckBERTScore,
    SelfCheckNLI,
)

app = FastAPI(title="HalluciGuard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

nlp = spacy.load("en_core_web_sm")

generator = pipeline(
    "text-generation",
    model="gpt2-medium",
    device=0 if torch.cuda.is_available() else -1,
)

selfcheck_bert = SelfCheckBERTScore(rescale_with_baseline=True)
selfcheck_nli  = SelfCheckNLI(device=device)


# ── Schemas ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str
    n_samples: int = Field(default=5, ge=1, le=10)
    max_new_tokens: int = Field(default=200, ge=50, le=500)


class SentenceResult(BaseModel):
    sentence: str
    bertscore: float
    nli: float
    combined: float


class AnalyzeResponse(BaseModel):
    response: str
    sentences: List[str]
    sentence_results: List[SentenceResult]
    # per-method averages
    avg_bertscore: float
    avg_nli: float
    avg_combined: float
    # per-method sentence scores
    scores_bertscore: List[float]
    scores_nli: List[float]
    scores_combined: List[float]
    n_samples: int
    samples: List[str]
    device: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "gpt2": "loaded",
        "scorers": ["bertscore", "nli", "combined"],
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # 1. Generate main response (greedy / deterministic)
    main_out = generator(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        do_sample=False,
        pad_token_id=50256,
    )
    main_resp = main_out[0]["generated_text"]

    # 2. Generate stochastic samples
    samples = []
    for _ in range(req.n_samples):
        out = generator(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=50256,
        )
        samples.append(out[0]["generated_text"])

    # 3. Sentence split
    sentences = [
        s.text.strip()
        for s in nlp(main_resp).sents
        if len(s.text.strip()) > 3
    ]
    if not sentences:
        raise HTTPException(status_code=422, detail="Could not extract sentences.")

    # 4. Score with ALL methods
    raw_bert = selfcheck_bert.predict(sentences, samples)
    raw_nli  = selfcheck_nli.predict(sentences, samples)

    scores_bert     = [round(float(s), 4) for s in raw_bert]
    scores_nli      = [round(float(s), 4) for s in raw_nli]
    # Combined = average of both
    scores_combined = [round((b + n) / 2, 4) for b, n in zip(scores_bert, scores_nli)]

    def avg(lst): return round(sum(lst) / len(lst), 4)

    sentence_results = [
        SentenceResult(
            sentence=s,
            bertscore=b,
            nli=n,
            combined=c,
        )
        for s, b, n, c in zip(sentences, scores_bert, scores_nli, scores_combined)
    ]

    return AnalyzeResponse(
        response=main_resp,
        sentences=sentences,
        sentence_results=sentence_results,
        avg_bertscore=avg(scores_bert),
        avg_nli=avg(scores_nli),
        avg_combined=avg(scores_combined),
        scores_bertscore=scores_bert,
        scores_nli=scores_nli,
        scores_combined=scores_combined,
        n_samples=req.n_samples,
        samples=samples,
        device=str(device),
    )
