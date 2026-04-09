from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task_eval import compute_task_metrics, extract_gold_answer_text, extract_gold_payload


def test_extract_gold_answer_text_supports_structured_quant_and_summary():
    assert extract_gold_answer_text(
        {"summary_one_sentence": "A concise summary.", "canonical_answer": "fallback"},
        "summary",
    ) == "A concise summary."
    assert extract_gold_answer_text(
        {"token": "Fig. 1", "quant_statement": "92% accuracy.", "canonical_answer": "Key statement: 92% accuracy. First figure/table: Fig. 1"},
        "quant_plus_first_figure",
    ) == "Key statement: 92% accuracy. First figure/table: Fig. 1"
    assert extract_gold_answer_text(
        {"canonical_answer": "2767 studies."},
        "chart_numeric",
    ) == "2767 studies."


def test_extract_gold_payload_supports_legacy_and_structured_schema():
    assert extract_gold_payload({"answer": "legacy answer"}) == "legacy answer"
    assert extract_gold_payload({"gold": {"token": "Fig. 1"}}) == {"token": "Fig. 1"}


def test_compute_task_metrics_aggregates_per_task_scores():
    preds = [
        {"question_type": "title", "pred_answer": "A Paper Title", "gold_answer": "A Paper Title"},
        {"question_type": "first_figure_token", "pred_answer": "Fig. 1", "gold_answer": "Fig. 1"},
        {"question_type": "summary", "pred_answer": "Here, we show a strong gain in performance.", "gold_answer": "This work shows a strong gain in performance."},
        {
            "question_type": "quant_plus_first_figure",
            "pred_answer": "Key statement: The method achieved 92% accuracy across 120 samples. First figure/table: Fig. 1",
            "gold_answer": {
                "token": "Fig. 1",
                "quant_statement": "The method achieved 92% accuracy across 120 samples.",
                "canonical_answer": "Key statement: The method achieved 92% accuracy across 120 samples. First figure/table: Fig. 1",
            },
        },
        {"question_type": "chart_numeric", "pred_answer": "2767 studies.", "gold_answer": {"canonical_answer": "2767 studies."}},
    ]
    metrics = compute_task_metrics(preds)
    assert metrics["title_em"] == 1.0
    assert metrics["first_figure_token_em"] == 1.0
    assert metrics["summary_token_f1"] > 0.4
    assert metrics["quant_token_match"] == 1.0
    assert metrics["quant_token_f1"] == 1.0
    assert metrics["quant_canonical_answer_em"] == 1.0
    assert metrics["chart_numeric_em"] == 1.0
