"""
Semantic entropy HTTP service for HASHIRU (Option B – Service 1).
Run in the semantic_uncertainty conda env. PYTHONPATH must include the directory
that contains the 'uncertainty' package (the inner semantic_uncertainty folder).

  cd .../semantic_uncertainty/semantic_uncertainty
  export PYTHONPATH="$(pwd)"
  uvicorn entropy_service:app --host 127.0.0.1 --port 8124

Expects POST /score with body: { "prompt", "responses", "samples?", "metadata?" }.
Returns { "entropy": [float], "clusters": [...], "reasons": {} }.

The "loading weights" message refers to the NLI model (DeBERTa), not your agent/worker LLM.
Set STRICT_ENTAILMENT=false to use loose clustering (more clusters, less often entropy=0).
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

# Reduce HuggingFace/httpx log noise (404s for optional files like adapter_config.json are normal)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Requires PYTHONPATH to include this directory (the one containing "uncertainty/")
from uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    get_semantic_ids,
    cluster_assignment_entropy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Entropy Service")

# Load model once at startup (DeBERTa; for LLM entailment you'd switch to EntailmentGPT4 etc.)
_entailment_model = None


def get_model():
    global _entailment_model
    if _entailment_model is None:
        logger.info("Loading entailment model (DeBERTa)...")
        _entailment_model = EntailmentDeberta()
        logger.info("Entailment model loaded.")
    return _entailment_model


class ScoreRequest(BaseModel):
    prompt: str
    responses: List[str] = Field(..., min_length=1)
    samples: Optional[List[List[str]]] = None
    metadata: Optional[Dict[str, Any]] = None


class EntropyScoreResponse(BaseModel):
    entropy: List[float]
    clusters: Optional[List[Any]] = None
    reasons: Dict[str, Any] = {}


def _truncate_for_entailment(text: str, max_words: int = 120) -> str:
    """Truncate to first max_words for DeBERTa's 512-token limit."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _last_n_words(text: str, n: int = 100) -> str:
    """Take the last n words (conclusions/verdicts often differ more than intros)."""
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[-n:])


def _prepend_question(question: str, answer_chunk: str, question_max_words: int = 40) -> str:
    """Prepend truncated question so NLI is question-conditioned (paper: entailment in context of the question)."""
    q_words = question.split()[:question_max_words]
    q_prefix = " ".join(q_words) if q_words else ""
    if not q_prefix:
        return answer_chunk
    return f"Question: {q_prefix}. Answer: {answer_chunk}"


@app.post("/score", response_model=EntropyScoreResponse)
def score(req: ScoreRequest) -> EntropyScoreResponse:
    if req.samples is not None and len(req.samples) != len(req.responses):
        raise HTTPException(
            status_code=422,
            detail=f"samples length ({len(req.samples)}) must match responses length ({len(req.responses)})",
        )
    n_resp = len(req.responses)
    n_samples_per = [len(req.samples[i]) if req.samples else 0 for i in range(n_resp)]
    logger.info("entropy request: %s response(s), samples per response: %s", n_resp, n_samples_per)
    model = get_model()
    entropies: List[float] = []
    all_clusters: List[Any] = []
    for i, resp in enumerate(req.responses):
        samples_i = req.samples[i] if req.samples else []
        all_responses = [resp] + list(samples_i)
        n_strings = len(all_responses)
        if n_strings < 2:
            logger.info("entropy item %s: only %s string(s), need ≥2 → entropy=0.0", i, n_strings)
            entropies.append(0.0)
            all_clusters.append([])
            continue
        last_n = int(os.environ.get("ENTROPY_LAST_N_WORDS", "100"))
        truncated = [_last_n_words(r, n=last_n) for r in all_responses]
        with_question = [_prepend_question(req.prompt, t) for t in truncated]
        word_lens = [len(s.split()) for s in with_question]
        strict_entailment = os.environ.get("STRICT_ENTAILMENT", "true").lower() in ("1", "true", "yes")
        raw_threshold = os.environ.get("ENTAILMENT_THRESHOLD", "").strip()
        entailment_threshold = float(raw_threshold) if raw_threshold else None
        logger.info("entropy item %s: %s strings (question + last %s words); word counts: %s; strict_entailment=%s; entailment_threshold=%s", i, n_strings, last_n, word_lens, strict_entailment, entailment_threshold)
        try:
            semantic_ids = get_semantic_ids(
                with_question,
                model=model,
                strict_entailment=strict_entailment,
                example=None,
                entailment_threshold=entailment_threshold,
            )
            ent = float(cluster_assignment_entropy(semantic_ids))
            ent = max(0.0, ent)
            n_clusters = len(set(semantic_ids))
            logger.info("entropy item %s: semantic_ids=%s → %s cluster(s), entropy=%.4f", i, semantic_ids, n_clusters, ent)
            entropies.append(ent)
            all_clusters.append(semantic_ids)
        except Exception as e:
            logger.exception("entropy item %s: computation failed: %s", i, e)
            entropies.append(0.0)
            all_clusters.append([])
    logger.info("entropy response: entropies=%s", entropies)
    return EntropyScoreResponse(
        entropy=entropies,
        clusters=all_clusters,
        reasons={"n": len(req.responses)},
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "entropy"}

