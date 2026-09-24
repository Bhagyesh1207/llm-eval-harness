from evalharness.metrics import best_gold_recall, is_correct, is_refusal


def test_refusal_detection():
    assert is_refusal("Not in context.")
    assert is_refusal("The context does not mention who did it.")
    assert not is_refusal("The Normans were in Normandy.")


def test_gold_recall_ignores_case_and_articles():
    assert best_gold_recall("They studied it in the Laboratory.", ["the laboratory"]) == 1.0
    assert best_gold_recall("No idea", ["laboratory"]) == 0.0


def test_correctness_rules():
    assert is_correct("Stephen Greenblatt teaches there.", ["Stephen Greenblatt"], True)
    assert not is_correct("E. O. Wilson teaches there.", ["Stephen Greenblatt"], True)
    assert is_correct("Not in context.", [], False)
    assert not is_correct("It was Telepad.", [], False)
