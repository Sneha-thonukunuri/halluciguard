"""
HalluciGuard — FastAPI backend
Runs GPT-2 generation + SelfCheckGPT hallucination scoring
"""

import torch
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
from transformers import pipeline

from selfcheckgpt.modeling_selfcheck import (
    SelfCheckBERTScore,
    SelfCheckNLI,
    SelfCheckMQAG,
)

app = FastAPI(title="HalluciGuard API", version="1.0.0")

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
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1,
)

selfcheck_bert = SelfCheckBERTScore(rescale_with_baseline=True)
selfcheck_nli  = SelfCheckNLI(device=device)
# SelfCheckMQAG is heavy — uncomment if you have enough RAM/VRAM
# selfcheck_mqag = SelfCheckMQAG(device=device)


# ── Request / Response schemas ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str
    n_samples: int = Field(default=5, ge=1, le=10)
    max_new_tokens: int = Field(default=200, ge=50, le=500)
    method: Literal["bertscore", "nli", "mqag"] = "bertscore"


class SentenceResult(BaseModel):
    sentence: str
    score: float


class AnalyzeResponse(BaseModel):
    response: str
    sentences: List[str]
    scores: List[float]
    sentence_results: List[SentenceResult]
    avg_score: float
    method: str
    n_samples: int
    samples: List[str]
    device: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "gpt2": "loaded",
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

    # 2. Generate stochastic samples for consistency check
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

    # 3. Split main response into sentences with spaCy
    sentences = [
        s.text.strip()
        for s in nlp(main_resp).sents
        if len(s.text.strip()) > 3
    ]

    if not sentences:
        raise HTTPException(status_code=422, detail="Could not extract sentences from generated text.")

    # 4. Score with selected SelfCheckGPT method
    if req.method == "bertscore":
        raw_scores = selfcheck_bert.predict(sentences, samples)

    elif req.method == "nli":
        raw_scores = selfcheck_nli.predict(sentences, samples)

    elif req.method == "mqag":
        # Uncomment selfcheck_mqag above to enable this
        raise HTTPException(status_code=400, detail="MQAG is disabled by default. Uncomment selfcheck_mqag in app.py to enable it.")

    scores = [round(float(s), 4) for s in raw_scores]
    avg_score = round(sum(scores) / len(scores), 4)

    sentence_results = [
        SentenceResult(sentence=s, score=sc)
        for s, sc in zip(sentences, scores)
    ]

    return AnalyzeResponse(
        response=main_resp,
        sentences=sentences,
        scores=scores,
        sentence_results=sentence_results,
        avg_score=avg_score,
        method=req.method,
        n_samples=req.n_samples,
        samples=samples,
        device=str(device),
    )
