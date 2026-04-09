import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doc_aware import (
    derive_doc_id_from_pdf_path,
    filter_ids_by_doc_id,
    is_clean_text_for_embed,
    normalize_doc_id,
    parse_doc_id_from_question,
    resolve_question_type,
)


def test_parse_doc_id_from_bracketed_question():
    doc_id, q = parse_doc_id_from_question("[sciadv.ado9413.PMC466938] What is the exact full paper title?")
    assert doc_id == "sciadv.ado9413.pmc466938"
    assert q == "What is the exact full paper title?"


def test_passage_candidate_filter_by_doc_id_metadata():
    hash_ids = ["p1", "p2", "p3"]
    metadata = {
        "p1": {"doc_id": "doc_a"},
        "p2": {"doc_id": "doc_b"},
        "p3": {"doc_id": "doc_a"},
    }
    out = filter_ids_by_doc_id(hash_ids, metadata, "doc_a")
    assert out == ["p1", "p3"]


def test_image_candidate_filter_by_doc_id_metadata():
    hash_ids = ["i1", "i2", "i3"]
    metadata = {
        "i1": {"doc_id": "doc_x"},
        "i2": {"doc_id": "doc_y"},
        "i3": {"doc_id": "doc_x"},
    }
    out = filter_ids_by_doc_id(hash_ids, metadata, "doc_x")
    assert out == ["i1", "i3"]


def test_embeddings_text_excludes_metadata_prefix():
    text_for_embed = "This is clean semantic text for embedding."
    display_text = "[PDF_META doc=doc_a | page=1 | type=text] This is clean semantic text for embedding."
    assert is_clean_text_for_embed(text_for_embed)
    assert not is_clean_text_for_embed(display_text)


def test_normalize_doc_id_variants_resolve_to_same_id():
    expected = "sciadv.adp1439.pmc466956"
    variants = [
        "sciadv.adp1439.PMC466956",
        "SCIADV.ADP1439.PMC466956.PDF",
        "SciAdv.Adp1439.Pmc466956.pdf",
        r"E:\dataset\pmc_pdf\PMC466956\sciadv.adp1439.PMC466956.pdf",
        "[sciadv.adp1439.PMC466956]",
    ]
    for v in variants:
        assert normalize_doc_id(v) == expected
    assert derive_doc_id_from_pdf_path(r"E:\dataset\pmc_pdf\PMC466956\sciadv.adp1439.PMC466956.pdf") == expected


def test_resolve_question_type_prefers_dataset_label():
    question_info = {
        "question": "Please answer freely.",
        "question_type": "title",
    }
    assert resolve_question_type(question_info, question_info["question"]) == "title"
    assert resolve_question_type({"question": "According to Fig. 2, how many studies were identified?", "question_type": "chart_numeric"}, "According to Fig. 2, how many studies were identified?") == "chart_numeric"


def test_resolve_question_type_falls_back_when_missing_or_empty():
    raw_question = "What is the exact full paper title?"
    assert resolve_question_type({}, raw_question) == "title"
    assert resolve_question_type({"question_type": ""}, raw_question) == "title"


def test_resolve_question_type_logs_and_falls_back_for_invalid_label(caplog):
    raw_question = "What is the exact full paper title?"
    with caplog.at_level("WARNING"):
        resolved = resolve_question_type({"question_type": "numeric"}, raw_question, logger=logging.getLogger("test"))
    assert resolved == "title"
    assert "question_type fallback reason=invalid" in caplog.text
