"""Get answers from any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM or a free tier).

Optional. The published results were generated locally with open models, see generate/.
"""
import json
import os
import time

import requests

PROMPTS = {
    "plain": "You are a helpful assistant. Answer the question in one short sentence.",
    "grounded": ('Answer the question using only the context. If the context does not contain '
                 'the answer, reply exactly "Not in context." Keep it to one short sentence.'),
}


def ask(base_url: str, model: str, system: str, context: str, question: str, api_key: str | None = None) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"model": model, "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]}
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def run(dataset: list[dict], base_url: str, model: str, out_path: str, prompts=("plain", "grounded")):
    key = os.environ.get("OPENAI_API_KEY")
    with open(out_path, "a") as f:
        for p in prompts:
            for it in dataset:
                t0 = time.time()
                a = ask(base_url, model, PROMPTS[p], it["context"], it["question"], key)
                f.write(json.dumps({"model": model, "prompt": p, "id": it["id"], "answer": a,
                                    "latency_s": round(time.time() - t0, 3)}) + "\n")
