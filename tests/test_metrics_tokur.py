"""Tests for TokUR-style detection metrics."""

from panda.metrics_tokur import compute_detection_metrics, top_p_accuracy


def test_top_p_accuracy_takes_most_confident_half():
    # Perfect UE: low score = correct, high score = wrong
    labels = [True, True, True, True, False, False, False, False]
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    acc = top_p_accuracy(labels, scores, p=0.5)
    assert acc == 1.0


def test_top_p_accuracy_inverted_sort_was_wrong():
    labels = [True, True, False, False]
    scores = [0.1, 0.2, 0.8, 0.9]
    # Old bug (desc on uncertainty): top 50% = wrong half -> 0.0
    import pandas as pd

    df = pd.DataFrame({"label": labels, "score": scores}).sort_values("score", ascending=False)
    assert float(df.iloc[:2]["label"].mean()) == 0.0
    # Fixed: lowest uncertainty half -> both correct -> 1.0
    assert top_p_accuracy(labels, scores, p=0.5) == 1.0


def test_compute_detection_metrics_acc_star_matches_confident_half():
    labels_correct = [True, True, False, False]
    scores = [0.1, 0.2, 0.8, 0.9]
    m = compute_detection_metrics(labels_correct, scores)
    assert m.acc_star == 1.0
    assert m.auroc > 0.5
