"""Usage:
  python -m evalharness judge   --dataset data/dataset.json --answers results/answers.jsonl --out results/judged.json
  python -m evalharness report  --judged results/judged.json --out results/leaderboard.md
  python -m evalharness generate --dataset data/dataset.json --base-url http://localhost:11434/v1 --model llama3.2 --out results/answers.jsonl
"""
import argparse
import json

from . import metrics


def judge(dataset_path, answers_path, out_path, entail_threshold=0.5):
    from . import judge as J  # loads the local judge models only when needed
    items = {i["id"]: i for i in json.load(open(dataset_path))}
    rows = [json.loads(l) for l in open(answers_path) if l.strip()]
    out = []
    for r in rows:
        it = items[r["id"]]
        refused = metrics.is_refusal(r["answer"])
        nli = J.faithfulness(it["context"], r["answer"])
        out.append({**r,
                    "refused": refused,
                    "entail": round(nli["entailment"], 3),
                    "contra": round(nli["contradiction"], 3),
                    "faithful": None if refused else nli["entailment"] >= entail_threshold,
                    "relevance": round(J.relevance(it["question"], r["answer"]), 3),
                    "gold_recall": round(metrics.best_gold_recall(r["answer"], it["gold"]), 2) if it["answerable"] else None,
                    "correct": metrics.is_correct(r["answer"], it["gold"], it["answerable"]),
                    # hallucinated = gave an answer the passage does not back up
                    "hallucinated": (not refused) and ((not it["answerable"]) or nli["entailment"] < entail_threshold)})
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def summarize(judged, dataset):
    items = {i["id"]: i for i in dataset}
    groups = {}
    for r in judged:
        groups.setdefault((r["model"], r["prompt"]), []).append(r)
    board = []
    for (m, p), rs in groups.items():
        ans = [r for r in rs if items[r["id"]]["answerable"]]
        una = [r for r in rs if not items[r["id"]]["answerable"]]
        given = [r for r in rs if not r["refused"]]
        pct = lambda xs, f: round(100 * sum(f(x) for x in xs) / len(xs), 1) if xs else None
        row = {"model": m, "prompt": p, "n": len(rs),
               "accuracy": pct(rs, lambda r: r["correct"]),
               "answerable_acc": pct(ans, lambda r: r["correct"]),
               "abstain_when_no_answer": pct(una, lambda r: r["refused"]),
               "faithfulness": pct(given, lambda r: r["faithful"]),
               "hallucination_rate": pct(rs, lambda r: r["hallucinated"]),
               "relevance": round(sum(r["relevance"] for r in rs) / len(rs), 3),
               "avg_latency_s": round(sum(r.get("latency_s", 0) for r in rs) / len(rs), 2)}
        row["score"] = round((row["accuracy"] + row["faithfulness"] + (100 - row["hallucination_rate"])) / 3, 1)
        board.append(row)
    return sorted(board, key=lambda r: -r["score"])


def report(judged_path, dataset_path, out_path):
    board = summarize(json.load(open(judged_path)), json.load(open(dataset_path)))
    cols = ["model", "prompt", "score", "accuracy", "answerable_acc", "abstain_when_no_answer",
            "faithfulness", "hallucination_rate", "relevance", "avg_latency_s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in board:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    open(out_path, "w").write("\n".join(lines) + "\n")
    json.dump(board, open(out_path.replace(".md", ".json"), "w"), indent=1)
    return board


def main():
    ap = argparse.ArgumentParser(prog="evalharness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge"); j.add_argument("--dataset", required=True); j.add_argument("--answers", required=True); j.add_argument("--out", required=True)
    r = sub.add_parser("report"); r.add_argument("--judged", required=True); r.add_argument("--dataset", required=True); r.add_argument("--out", required=True)
    g = sub.add_parser("generate"); g.add_argument("--dataset", required=True); g.add_argument("--base-url", required=True); g.add_argument("--model", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "judge":
        print(f"judged {len(judge(a.dataset, a.answers, a.out))} answers -> {a.out}")
    elif a.cmd == "report":
        for row in report(a.judged, a.dataset, a.out):
            print(row)
    else:
        from .generate import run
        run(json.load(open(a.dataset)), a.base_url, a.model, a.out)


if __name__ == "__main__":
    main()
