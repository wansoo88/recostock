"""FinBERT polarity scorer — runs only if transformers + torch are installed.

Model: ProsusAI/finbert (3-class: positive / neutral / negative).
Polarity is collapsed to a single scalar in [-1, +1]:

    polarity = P(positive) - P(negative)

Design notes
------------
- Lazy import: the project's day-to-day code (collect_sentiment.py, local
  dev) does NOT require torch. `is_available()` reports honestly; when
  unavailable, `score_batch` returns Nones so the aggregator simply writes
  polarity_n=0 for that doc.
- Single model load per process. The GitHub Actions cache step keeps the
  ~/.cache/huggingface model weights warm between daily runs.
- Truncates to model max length (512 tokens) — Yahoo headlines and HN
  titles are well under this; EDGAR doc bodies are short cashtag strings.
"""
from __future__ import annotations

import logging
from typing import Iterable

log = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_LEN = 256
BATCH_SIZE = 16

# Populated on first successful load.
_tokenizer = None
_model = None
_torch = None
_load_attempted = False
_load_failed_reason: str | None = None


def is_available() -> bool:
    """Return True iff transformers + torch + the FinBERT weights load cleanly."""
    _try_load()
    return _model is not None


def _try_load() -> None:
    global _tokenizer, _model, _torch, _load_attempted, _load_failed_reason
    if _load_attempted:
        return
    _load_attempted = True
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        _load_failed_reason = f"missing dependency: {exc}"
        log.info("FinBERT scorer disabled — %s", _load_failed_reason)
        return
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        mdl.eval()
    except Exception as exc:
        _load_failed_reason = f"weights load failed: {exc}"
        log.warning("FinBERT scorer disabled — %s", _load_failed_reason)
        return
    _tokenizer, _model, _torch = tok, mdl, torch
    log.info("FinBERT loaded: %s", MODEL_NAME)


def _label_to_index() -> dict[str, int]:
    """Locate positive/negative columns regardless of model's label ordering."""
    cfg = _model.config
    id2label = {int(i): str(l).lower() for i, l in cfg.id2label.items()}
    return {label: idx for idx, label in id2label.items()}


def score_batch(texts: list[str]) -> list[float | None]:
    """Return polarity ∈ [-1, +1] (pos - neg) per text, or None if unavailable."""
    _try_load()
    if _model is None:
        return [None] * len(texts)

    clean = [t.strip() for t in texts]
    keep_mask = [bool(t) for t in clean]
    if not any(keep_mask):
        return [None] * len(texts)

    label_idx = _label_to_index()
    pos_idx = label_idx.get("positive")
    neg_idx = label_idx.get("negative")
    if pos_idx is None or neg_idx is None:
        log.warning("FinBERT label set unexpected: %s — disabling", label_idx)
        return [None] * len(texts)

    out: list[float | None] = [None] * len(texts)
    indices = [i for i, k in enumerate(keep_mask) if k]
    payload = [clean[i] for i in indices]

    with _torch.no_grad():
        for start in range(0, len(payload), BATCH_SIZE):
            chunk = payload[start:start + BATCH_SIZE]
            enc = _tokenizer(
                chunk, padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt",
            )
            logits = _model(**enc).logits
            probs = _torch.softmax(logits, dim=-1).cpu().numpy()
            for local_i, doc_i in enumerate(indices[start:start + BATCH_SIZE]):
                pol = float(probs[local_i, pos_idx] - probs[local_i, neg_idx])
                out[doc_i] = pol
    return out


def score_documents(docs: Iterable[dict], text_keys: tuple[str, ...] = ("title", "body")) -> list[float | None]:
    """Score a stream of doc dicts; uses concatenated text_keys per doc."""
    docs = list(docs)
    texts = [" ".join(filter(None, (d.get(k, "") for k in text_keys))) for d in docs]
    return score_batch(texts)
