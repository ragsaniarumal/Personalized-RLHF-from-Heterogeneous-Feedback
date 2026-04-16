from hetero_rlhf.data import parse_openai_comparison, balanced_subset


def test_parse_summary_choice_schema():
    row = {
        "worker": "w1",
        "info": {"post": "A long Reddit post"},
        "summaries": [{"text": "bad"}, {"text": "good"}],
        "choice": 1,
    }
    rec = parse_openai_comparison(row)
    assert rec.worker == "w1"
    assert rec.chosen == "good"
    assert rec.rejected == "bad"


def test_balanced_subset():
    records = [
        parse_openai_comparison({
            "worker": worker,
            "prompt": "p",
            "chosen": f"c{i}",
            "rejected": f"r{i}",
        })
        for worker in ("a", "b")
        for i in range(4)
    ]
    sample = balanced_subset(records, ["a", "b"], per_worker=2, seed=7)
    assert len(sample) == 4
    assert sum(r.worker == "a" for r in sample) == 2
    assert sum(r.worker == "b" for r in sample) == 2
