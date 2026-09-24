"""Local judge models (ONNX). Free, no API key, same scores every run.

Faithfulness: an NLI model checks whether the context entails the answer.
Relevance: cosine similarity between question and answer embeddings.
"""
import json
import re
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

NLI_REPO = "Xenova/nli-deberta-v3-xsmall"
EMB_REPO = "Xenova/all-MiniLM-L6-v2"


def _load(repo: str):
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    tok.enable_truncation(512)
    sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model_quantized.onnx"),
                                providers=["CPUExecutionProvider"])
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    return tok, sess, cfg


def _feed(sess, enc_list):
    n = max(len(e.ids) for e in enc_list)
    pad = lambda xs: [x + [0] * (n - len(x)) for x in xs]
    feed = {"input_ids": np.array(pad([e.ids for e in enc_list]), dtype=np.int64),
            "attention_mask": np.array(pad([e.attention_mask for e in enc_list]), dtype=np.int64)}
    names = {i.name for i in sess.get_inputs()}
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.array(pad([e.type_ids for e in enc_list]), dtype=np.int64)
    return feed


@lru_cache(maxsize=1)
def _nli():
    return _load(NLI_REPO)


@lru_cache(maxsize=1)
def _emb():
    return _load(EMB_REPO)


def nli_scores(premise: str, hypothesis: str) -> dict[str, float]:
    """Probabilities for entailment / neutral / contradiction."""
    tok, sess, cfg = _nli()
    logits = sess.run(None, _feed(sess, [tok.encode(premise, hypothesis)]))[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return {cfg["id2label"][str(i)].lower(): float(v) for i, v in enumerate(p)}


def embed(texts: list[str]) -> np.ndarray:
    tok, sess, _ = _emb()
    encs = [tok.encode(t) for t in texts]
    feed = _feed(sess, encs)
    out = sess.run(None, feed)[0]
    mask = feed["attention_mask"][..., None]
    v = (out * mask).sum(1) / mask.sum(1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def relevance(question: str, answer: str) -> float:
    q, a = embed([question, answer])
    return float(q @ a)


def _windows(context: str) -> list[str]:
    """Whole passage, each sentence, and each pair of neighbouring sentences."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", context) if x.strip()]
    wins = [context] + sents + [" ".join(sents[i:i + 2]) for i in range(len(sents) - 1)]
    return list(dict.fromkeys(wins))


def faithfulness(context: str, answer: str) -> dict[str, float]:
    """Best NLI scores over passage windows. Small NLI models read short premises
    much better than a whole paragraph, so each claim is checked against the
    sentence(s) that could support it (same idea as SummaC)."""
    best = None
    for w in _windows(context):
        sc = nli_scores(w, answer)
        if best is None or sc["entailment"] > best["entailment"]:
            best = sc
    return best
