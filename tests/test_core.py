"""Smoke tests: python -m pytest -q"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import DATA_PATH, MODEL_DIR, Scorer, prepare, validate  # noqa: E402

scorer = Scorer()
data = pd.read_csv(DATA_PATH)


def test_segments_match_notebook():
    seg = scorer.segment(prepare(data))
    assert (seg.Segment.values == data.Segment_Name.values).mean() == 1.0


def test_metrics_match_report():
    m = json.loads((MODEL_DIR / "metadata.json").read_text())["rf_test_metrics"]
    assert abs(m["ROC_AUC"] - 0.9361) < 1e-3 and abs(m["Recall"] - 0.80) < 1e-3


def test_sample_scores_and_validates():
    s = pd.read_csv(Path(__file__).resolve().parents[1] / "sample" / "sample_customers.csv")
    assert validate(s) == []
    out = scorer.score(s)
    assert len(out) == len(s) and out.Default_Probability_RF.between(0, 1).all()


def test_missing_column_is_reported():
    s = pd.read_csv(Path(__file__).resolve().parents[1] / "sample" / "sample_customers.csv").drop(columns="Credit_Score")
    assert any("Credit_Score" in p for p in validate(s))
