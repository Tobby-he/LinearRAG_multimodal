import json
from pathlib import Path


def test_dev_set_exists_and_has_12_questions():
    p = Path(__file__).resolve().parents[1] / "dataset" / "pdf_test" / "questions_dev_12.json"
    assert p.exists(), f"missing dev set file: {p}"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 12
    for row in data:
        assert "question" in row and "answer" in row

