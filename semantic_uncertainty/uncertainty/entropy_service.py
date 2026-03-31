"""
Semantic entropy HTTP service for HASHIRU (Option B – Service 1).
Run in the semantic_uncertainty conda env. PYTHONPATH must include the directory
that contains the 'uncertainty' package (the inner semantic_uncertainty folder).

  cd .../semantic_uncertainty/semantic_uncertainty
  export PYTHONPATH="$(pwd)"
  uvicorn entropy_service:app --host 127.0.0.1 --port 8124

Expects POST /score with body: { "prompt", "responses", "samples?", "metadata?" }.
Returns { "entropy": [float], "clusters": [...], "reasons": {} }.

Optional **metadata["sequence_logprobs"]**: one summed log-likelihood per string in
`[responses[i]] + samples[i]` (same order as the service). When all values are finite
and length matches, entropy uses predictive (Rao–Blackwell) semantic entropy
(`logsumexp_by_id` + `predictive_entropy_rao`). If missing, wrong length, or any
`null`/non-finite entry, the score falls back to cluster-assignment entropy only.

Set ENTROPY_USE_SEQUENCE_LOGPROBS=0 to always use cluster-assignment entropy.

The "loading weights" message refers to the NLI model (DeBERTa), not your agent/worker LLM.
STRICT_ENTAILMENT=false. ENTAILMENT_THRESHOLD=0.5. REQUIRE_ENTAILMENT_WINNER=at_least_one (default): merge only when at least one direction is entailment, so (1,1) and contradiction never merge; true=both entailment, false=allow (1,1).
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np

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
    logsumexp_by_id,
    predictive_entropy_rao,
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


def _env_use_sequence_logprobs() -> bool:
    v = os.environ.get("ENTROPY_USE_SEQUENCE_LOGPROBS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _parse_sequence_logprobs(
    metadata: Optional[Dict[str, Any]],
    item_index: int,
    n_strings: int,
    n_responses: int,
) -> Optional[List[float]]:
    """Return per-string log-likelihoods for this item, or None if unusable."""
    if not metadata:
        return None
    raw = metadata.get("sequence_logprobs")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], (list, tuple)):
        if item_index >= len(raw):
            return None
        seq = list(raw[item_index])
    else:
        if item_index != 0 and n_responses > 1:
            return None
        seq = list(raw)
    if len(seq) != n_strings:
        return None
    out: List[float] = []
    for x in seq:
        if x is None:
            return None
        try:
            f = float(x)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(f):
            return None
        out.append(f)
    return out


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
    entropy_modes: List[str] = []
    used_logprobs_flags: List[bool] = []
    use_lp = _env_use_sequence_logprobs()
    for i, resp in enumerate(req.responses):
        samples_i = req.samples[i] if req.samples else []
        all_responses = [resp] + list(samples_i)
        n_strings = len(all_responses)
        if n_strings < 2:
            logger.info("entropy item %s: only %s string(s), need ≥2 → entropy=0.0", i, n_strings)
            entropies.append(0.0)
            all_clusters.append([])
            entropy_modes.append("trivial_single")
            used_logprobs_flags.append(False)
            continue
        last_n = int(os.environ.get("ENTROPY_LAST_N_WORDS", "100"))
        truncated = [_last_n_words(r, n=last_n) for r in all_responses]
        with_question = [_prepend_question(req.prompt, t) for t in truncated]
        word_lens = [len(s.split()) for s in with_question]
        # Default false = loose clustering (equivalent = not contradiction and not both neutral); identical/paraphrases merge.
        strict_entailment = os.environ.get("STRICT_ENTAILMENT", "false").lower() in ("1", "true", "yes")
        raw_threshold = os.environ.get("ENTAILMENT_THRESHOLD", "").strip()
        # Default 0.5 with at_least_one: merge when one direction is entailment; (1,1) and contradiction never merge.
        default_threshold = 0.35
        entailment_threshold = float(raw_threshold) if raw_threshold else default_threshold
        raw_winner = os.environ.get("REQUIRE_ENTAILMENT_WINNER", "at_least_one").strip().lower()
        require_entailment_winner = True if raw_winner in ("1", "true", "yes") else (False if raw_winner in ("0", "false", "no") else "at_least_one")
        raw_neutral = os.environ.get("NEUTRAL_MERGE_THRESHOLD", "").strip()
        neutral_merge_threshold = float(raw_neutral) if raw_neutral else None
        logger.info("entropy item %s: %s strings; strict_entailment=%s; entailment_threshold=%s; require_entailment_winner=%s; neutral_merge_threshold=%s", i, n_strings, strict_entailment, entailment_threshold, require_entailment_winner, neutral_merge_threshold)
        try:
            semantic_ids = get_semantic_ids(
                with_question,
                model=model,
                strict_entailment=strict_entailment,
                example=None,
                entailment_threshold=entailment_threshold,
                require_entailment_winner=require_entailment_winner,
                neutral_merge_threshold=neutral_merge_threshold,
            )
            cluster_ent = float(cluster_assignment_entropy(semantic_ids))
            cluster_ent = max(0.0, cluster_ent)
            mode = "cluster_assignment"
            ent = cluster_ent
            used_lp_here = False
            seq_lp = None
            if use_lp:
                seq_lp = _parse_sequence_logprobs(
                    req.metadata, i, n_strings, n_resp
                )
            if seq_lp is not None:
                try:
                    log_lik_sem = logsumexp_by_id(
                        semantic_ids,
                        np.asarray(seq_lp, dtype=np.float64),
                        agg="sum_normalized",
                    )
                    ent = float(predictive_entropy_rao(log_lik_sem))
                    ent = max(0.0, ent)
                    mode = "predictive_rao"
                    used_lp_here = True
                except Exception as lp_exc:
                    logger.warning(
                        "entropy item %s: predictive_rao failed (%s), using cluster_assignment_entropy",
                        i,
                        lp_exc,
                    )
                    ent = cluster_ent
                    mode = "cluster_assignment_fallback"
            n_clusters = len(set(semantic_ids))
            logger.info(
                "entropy item %s: semantic_ids=%s → %s cluster(s), mode=%s, entropy=%.4f",
                i,
                semantic_ids,
                n_clusters,
                mode,
                ent,
            )
            entropies.append(ent)
            all_clusters.append(semantic_ids)
            entropy_modes.append(mode)
            used_logprobs_flags.append(used_lp_here)
        except Exception as e:
            logger.exception("entropy item %s: computation failed: %s", i, e)
            entropies.append(0.0)
            all_clusters.append([])
            entropy_modes.append("error")
            used_logprobs_flags.append(False)
    logger.info("entropy response: entropies=%s", entropies)
    return EntropyScoreResponse(
        entropy=entropies,
        clusters=all_clusters,
        reasons={
            "n": len(req.responses),
            "entropy_modes": entropy_modes,
            "used_sequence_logprobs": used_logprobs_flags,
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "entropy"}

