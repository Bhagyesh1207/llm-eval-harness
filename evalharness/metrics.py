"""Plain Python metrics. No model needed."""
import re

REFUSAL = re.compile(
    r"not in (the )?context|not (mentioned|provided|stated|specified|given)"
    r"|does not (contain|mention|say|provide|specify|state)"
    r"|doesn't (contain|mention|say|provide|specify)|no information"
    r"|cannot be determined|can't be determined|unable to (answer|determine)|is unknown",
    re.I,
)


def is_refusal(answer: str) -> bool:
    """True when the model says the context has no answer."""
    return bool(REFUSAL.search(answer))


def tokens(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def gold_recall(answer: str, gold: str) -> float:
    """Share of the gold answer's words that appear in the model answer."""
    g = tokens(gold)
    if not g:
        return 0.0
    a = set(tokens(answer))
    return sum(t in a for t in g) / len(g)


def best_gold_recall(answer: str, golds: list[str]) -> float:
    return max((gold_recall(answer, g) for g in golds), default=0.0)


def is_correct(answer: str, golds: list[str], answerable: bool, threshold: float = 0.6) -> bool:
    """Answerable: must contain most of a gold answer. Unanswerable: must refuse."""
    if not answerable:
        return is_refusal(answer)
    return (not is_refusal(answer)) and best_gold_recall(answer, golds) >= threshold
