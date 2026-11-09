from scripts.review_audio_shared_execution_v65 import bootstrap_review


def test_resampling_keeps_correlated_variants_in_their_foreground_group():
    rows = [{"group": "a", "label": 1, "decisions": {"current": "inconclusive", "combined": "ai"}}
            for _ in range(50)]
    rows += [{"group": "b", "label": 0, "decisions": {"current": "original", "combined": "inconclusive"}}
             for _ in range(2)]
    intervals = bootstrap_review(rows, {"seed": 1, "replicates": 1000})
    assert intervals == {"current": [0.0, 0.0], "combined": [-4.0, 100.0]}


def test_lost_correct_genuine_calls_reduce_useful_conclusions():
    rows = [{"group": "one", "label": 0, "decisions": {"current": "original", "combined": "ai"}}
            for _ in range(5)]
    assert bootstrap_review(rows, {"seed": 2, "replicates": 100})["combined"] == [-5.0, -5.0]
