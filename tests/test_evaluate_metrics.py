from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_metrics import exact_match_score, token_f1_score


def test_summary_token_f1_is_partial_match_friendly():
    score = token_f1_score(
        "Here, we introduce a bimodal actuation strategy for programmable 3D droplet manipulation.",
        "This work introduces a bimodal magnetic-optical actuation strategy for programmable three-dimensional manipulation of ferrofluidic droplets.",
    )
    assert 0.4 < score < 1.0


def test_exact_match_is_normalized():
    assert exact_match_score("Fig. 1", "fig 1") == 1.0
