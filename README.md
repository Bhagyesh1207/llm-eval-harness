# 🧪 LLM Eval Harness

<p>
  <a href="https://llm-eval-harness.vercel.app"><img src="https://img.shields.io/badge/%E2%96%B6%20Live%20demo-llm--eval--harness.vercel.app-FF3B2F?style=for-the-badge" alt="Live demo" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/API%20key-not%20required-2EA44F?style=for-the-badge" alt="No API key required" />
  <img src="https://img.shields.io/badge/License-MIT-555?style=for-the-badge" alt="MIT license" />
</p>

**Score LLM answers for correctness, faithfulness and hallucination, with a free judge that runs on your own machine.**

Live demo: **https://llm-eval-harness.vercel.app** (leaderboard, every answer with its passage, and a box to score your own answer in the browser)

Most teams find out a model hallucinates when a user complains. This harness gives a repeatable test instead: a fixed question set with source passages, answers from the models you want to compare, and a judge that checks each answer against its passage. It grew out of my work evaluating LLM output for accuracy and coherence, and my earlier [KPI validation project with DeepEval](https://github.com/Bhagyesh1207/Automated-KPI-Validation-with-Deep-Eval).

**No API key, no cost.** The models under test and the judge are all open models that run locally on CPU. Same inputs, same scores, every run.

## Results

40 questions from SQuAD v2, 12 of which have **no answer** in their passage (the hallucination trap). Two open models, two prompt styles, 160 answers in total.

| # | Model | Prompt | Score | Accuracy % | Says "no answer" % | Faithful % | Hallucinated % | Relevance |
|---|---|---|---|---|---|---|---|---|
| 1 | SmolLM2-360M | grounded | **64.4** | 45.0 | 25.0 | 75.8 | 27.5 | 0.663 |
| 2 | SmolLM2-360M | plain | **55.8** | 45.0 | 0.0 | 70.0 | 47.5 | 0.773 |
| 3 | SmolLM2-135M | plain | **46.7** | 25.0 | 0.0 | 67.5 | 52.5 | 0.684 |
| 4 | SmolLM2-135M | grounded | **35.7** | 17.5 | 8.3 | 47.1 | 57.5 | 0.575 |

- **Score** = average of accuracy, faithfulness and (100 minus hallucination rate).
- **plain**: "Answer the question in one short sentence." **grounded**: "Answer using only the context. If the context does not contain the answer, reply exactly \"Not in context.\""

**What it shows**

- On the same 360M model, the grounded prompt cut hallucinations from **47.5% to 27.5%**.
- Even with that prompt, the 360M model said "no answer" on only 25% of the questions that had none. The prompt alone is not enough of a guardrail for small models.
- The 135M model hallucinated on more than half its answers with either prompt, and the grounded prompt made it worse (twice it just repeated the instruction instead of answering). Smaller models do not follow instructions reliably.

Full per-answer scores: [`results/judged.json`](results/judged.json). Leaderboard: [`results/leaderboard.md`](results/leaderboard.md).

## How the judge works

| Metric | How it is measured |
|---|---|
| **Faithfulness** | An NLI model ([`nli-deberta-v3-xsmall`](https://huggingface.co/Xenova/nli-deberta-v3-xsmall)) checks whether the passage entails the answer. Small NLI models read short premises better than long ones, so each answer is checked against the whole passage, each sentence and each pair of neighbouring sentences, and the best score counts (same idea as SummaC). Faithful means entailment of at least 0.5. |
| **Correctness** | Answerable questions: the reply must contain at least 60% of the words of a gold answer. Unanswerable questions: the model must say the passage has no answer. |
| **Hallucination** | The model gave an answer the passage does not support, or answered a question the passage cannot answer. |
| **Relevance** | Cosine similarity of question and answer embeddings ([`all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2)). |

### Is the judge right?

An eval is only as good as its judge, so 20 random answers were checked by hand ([`results/hand_check.json`](results/hand_check.json)). The judge agreed on **14 of 20 (70%)**. It passed 5 answers that were not really supported (for example "Rin" when the passage says "Rhin") and missed 1 that was. So it leans lenient: read its hallucination numbers as a floor, and swap in a bigger judge for high-stakes use.

## Run it

```bash
pip install -r requirements.txt

# score answers (downloads the two small judge models, about 100 MB, once)
python -m evalharness judge  --dataset data/dataset.json --answers results/answers.jsonl --out results/judged.json
python -m evalharness report --judged results/judged.json --dataset data/dataset.json --out results/leaderboard.md
```

Test your own model through any OpenAI-compatible endpoint, for example [Ollama](https://ollama.com) running locally (free):

```bash
python -m evalharness generate --dataset data/dataset.json --base-url http://localhost:11434/v1 --model llama3.2 --out results/answers.jsonl
```

Then run `judge` and `report` again and your model shows up on the leaderboard. Set `OPENAI_API_KEY` only if your endpoint needs one.

The published answers were generated on CPU with [transformers.js](https://huggingface.co/docs/transformers.js) (greedy decoding, 48 new tokens):

```bash
cd generate && npm install && node local_models.mjs HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-135M-Instruct
```

Tests: `python -m pytest tests`

## Project layout

```
evalharness/   metrics.py (refusals, gold match), judge.py (local NLI + embeddings), generate.py, cli.py
data/          dataset.json: 40 SQuAD v2 questions with passages and gold answers
results/       answers.jsonl, judged.json, leaderboard.md/json, hand_check.json
generate/      local_models.mjs: produces answers with open models, no key
demo/          index.html: the live demo (static, judge runs in the browser)
tests/         unit tests for the metrics
```

## Limits

- 40 questions is a small set. It is enough to show clear gaps between runs, not to rank close models.
- The judge is small and lenient (see the hand check). A larger NLI model or an LLM judge would be stricter.
- Correctness uses word overlap with the gold answer, so a correct answer worded very differently can be marked wrong.

## Credits

Questions from [SQuAD v2](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0). Models: SmolLM2 by Hugging Face, judge models via Xenova on the Hugging Face Hub.

Built by [Bhagyesh Patel](https://itsbhagyesh.vercel.app) · [LinkedIn](https://linkedin.com/in/13hagyesh)
