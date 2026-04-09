import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_mpdocvqa


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def make_local_tmp_dir(name: str) -> Path:
    base = ROOT / "test_tmp_artifacts"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_normalize_question_row_with_explicit_doc_and_flat_answer():
    row = {
        "questionId": "q1",
        "question": "What is asked?",
        "docId": "DocA.PDF",
        "answer": "Answer A",
        "split": "train",
    }
    normalized = run_mpdocvqa.normalize_mpdocvqa_question_row(row, annotation_map={})
    assert normalized["question"] == "[doca] What is asked?"
    assert normalized["doc_id"] == "doca"
    assert normalized["answer"] == "Answer A"
    assert normalized["question_id"] == "q1"


def test_normalize_question_row_derives_doc_id_from_image_and_structured_gold():
    row = {
        "qid": "q2",
        "question_text": "Summarize this page",
        "image": "PMC123_page_0002.jpg",
    }
    annotation_map = {
        "q2": {
            "qid": "q2",
            "gold": {"canonical_answer": "Main answer"},
        }
    }
    normalized = run_mpdocvqa.normalize_mpdocvqa_question_row(row, annotation_map=annotation_map)
    assert normalized["doc_id"] == "pmc123_page_0002"
    assert normalized["gold"]["canonical_answer"] == "Main answer"


def test_clean_mpdocvqa_text_normalizes_unicode_quotes():
    text = "What is the ‘actual’ value per 1000, during the year 1975?"
    assert run_mpdocvqa.clean_mpdocvqa_text(text) == "What is the 'actual' value per 1000, during the year 1975?"


def test_extract_page_ids_parses_mpdocvqa_page_strings():
    row = {"page_ids": ["ffbf0023_p0", "ffbf0023_p7", "bad_value"]}
    assert run_mpdocvqa._extract_page_ids(row) == [0, 7]


def test_load_questions_filters_by_split_manifest():
    root = make_local_tmp_dir("mpdocvqa_split") / "mpdocvqa"
    write_json(
        root / "Questions JSON" / "questions.json",
        [
            {"questionId": "q1", "question": "Q1", "docId": "doc1"},
            {"questionId": "q2", "question": "Q2", "docId": "doc2"},
        ],
    )
    (root / "Train Test Validation splits").mkdir(parents=True)
    (root / "Train Test Validation splits" / "train.txt").write_text("q1\n", encoding="utf-8")
    questions = run_mpdocvqa.load_mpdocvqa_questions(root, "train")
    assert [q["question_id"] for q in questions] == ["q1"]


def test_load_questions_supports_qas_directory_name():
    root = make_local_tmp_dir("mpdocvqa_qas") / "mpdocvqa"
    write_json(
        root / "qas" / "val.json",
        {
            "dataset_name": "MP-DocVQA",
            "dataset_split": "val",
            "data": [
                {
                    "questionId": "q1",
                    "question": "What is shown?",
                    "doc_id": "ffbf0023",
                    "page_ids": ["ffbf0023_p0", "ffbf0023_p1"],
                    "data_split": "val",
                }
            ],
        },
    )
    questions = run_mpdocvqa.load_mpdocvqa_questions(root, "val")
    assert len(questions) == 1
    assert questions[0]["doc_id"] == "ffbf0023"
    assert questions[0]["page_ids"] == [0, 1]


def test_asset_resolution_prefers_pdf_and_keeps_images():
    tmp_path = make_local_tmp_dir("mpdocvqa_assets")
    pdf_dir = tmp_path / "Images_PDFs"
    img_dir = tmp_path / "Images_PDFs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "doc1.pdf").write_text("pdf", encoding="utf-8")
    (img_dir / "doc1_page_0001.jpg").write_text("img", encoding="utf-8")
    asset_index = run_mpdocvqa.build_asset_index([pdf_dir], [img_dir])
    pdf_path, image_paths = run_mpdocvqa.resolve_question_assets(
        {"doc_id": "doc1", "source_image": "doc1_page_0001.jpg"},
        asset_index,
    )
    assert pdf_path is not None and pdf_path.name == "doc1.pdf"
    assert len(image_paths) == 1 and image_paths[0].name == "doc1_page_0001.jpg"


def test_build_corpus_uses_image_fallback_when_pdf_missing():
    tmp_path = make_local_tmp_dir("mpdocvqa_fallback")
    img_dir = tmp_path / "Images"
    img_dir.mkdir(parents=True)
    (img_dir / "doc2_page_0001.jpg").write_text("img", encoding="utf-8")
    asset_index = run_mpdocvqa.build_asset_index([tmp_path], [img_dir])
    questions = [{"doc_id": "doc2", "question": "[doc2] What?"}]
    passages, images, pdf_docs, registry, errors = run_mpdocvqa.build_corpus_for_questions(questions, asset_index)
    assert pdf_docs == []
    assert len(images) == 1
    assert len(passages) == 1
    assert registry["doc2"]["source"] == "images"
    assert errors == []


def test_build_corpus_uses_ocr_fallback_when_pdf_and_images_missing():
    tmp_path = make_local_tmp_dir("mpdocvqa_ocr")
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir(parents=True)
    write_json(
        ocr_dir / "ffbf0023_p0.json",
        {
            "LINE": [
                {"Text": "R. J. REYNOLDS TOBACCO COMPANY"},
                {"Text": "RETAIL PARTNERS MARKETING PLAN CONTRACT"},
            ]
        },
    )
    write_json(
        ocr_dir / "ffbf0023_p1.json",
        {
            "LINE": [
                {"Text": "Cigarettes represent less than 80% of ACV."},
            ]
        },
    )
    asset_index = run_mpdocvqa.build_asset_index([tmp_path], [tmp_path], [ocr_dir])
    questions = [{"doc_id": "ffbf0023", "question": "[ffbf0023] What?"}]
    passages, images, pdf_docs, registry, errors = run_mpdocvqa.build_corpus_for_questions(questions, asset_index)
    assert pdf_docs == []
    assert images == []
    assert len(passages) == 2
    assert passages[0]["block_type"] == "ocr_page"
    assert passages[0]["page"] == 0
    assert "R. J. REYNOLDS" in passages[0]["text_for_embed"]
    assert registry["ffbf0023"]["source"] == "ocr"
    assert errors == []


def test_build_ocr_passages_cleans_unicode_punctuation():
    tmp_path = make_local_tmp_dir("mpdocvqa_ocr_clean")
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir(parents=True)
    write_json(
        ocr_dir / "docx_p0.json",
        {"LINE": [{"Text": "What is the ‘actual’ value?"}]},
    )
    passages = run_mpdocvqa.build_ocr_passages("docx", [ocr_dir / "docx_p0.json"])
    assert len(passages) == 1
    assert passages[0]["text_for_embed"] == "What is the 'actual' value?"


def test_build_corpus_keeps_images_when_ocr_exists():
    tmp_path = make_local_tmp_dir("mpdocvqa_ocr_images")
    ocr_dir = tmp_path / "ocr"
    img_dir = tmp_path / "images"
    ocr_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    write_json(
        ocr_dir / "ffbf0023_p0.json",
        {"LINE": [{"Text": "R. J. REYNOLDS TOBACCO COMPANY"}]},
    )
    (img_dir / "ffbf0023_p0.jpg").write_text("img", encoding="utf-8")
    asset_index = run_mpdocvqa.build_asset_index([tmp_path], [img_dir], [ocr_dir])
    questions = [{"doc_id": "ffbf0023", "question": "[ffbf0023] What?"}]
    passages, images, pdf_docs, registry, errors = run_mpdocvqa.build_corpus_for_questions(questions, asset_index)
    assert pdf_docs == []
    assert len(passages) == 1
    assert len(images) == 1
    assert images[0]["doc_id"] == "ffbf0023"
    assert registry["ffbf0023"]["source"] == "ocr+images"
    assert registry["ffbf0023"]["image_count"] == 1
    assert errors == []


def test_build_error_prediction_is_eval_compatible():
    row = {
        "question": "[docx] What is this?",
        "doc_id": "docx",
        "question_type": "general",
        "answer": "gold",
        "question_id": "qid-1",
    }
    result = run_mpdocvqa.build_error_prediction(row, reason="missing_document", error="not found")
    assert result["gold_answer"] == "gold"
    assert result["used_route"] == "error"
    assert result["fallback_used"] is True
    assert result["question_id"] == "qid-1"


def test_main_writes_partial_predictions_and_continues_on_question_error(monkeypatch):
    tmp_path = make_local_tmp_dir("mpdocvqa_main")
    root = tmp_path / "mpdocvqa"
    write_json(
        root / "Questions JSON" / "questions_train.json",
        [
            {"questionId": "ok-q", "question": "What is shown?", "docId": "doc1", "answer": "ok"},
            {"questionId": "bad-q", "question": "Broken question?", "docId": "doc1", "answer": "bad"},
        ],
    )
    (root / "Train Test Validation splits").mkdir(parents=True)
    (root / "Train Test Validation splits" / "train.txt").write_text("ok-q\nbad-q\n", encoding="utf-8")
    pdf_dir = root / "Images PDFs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "doc1.pdf").write_text("pdf", encoding="utf-8")

    def fake_parse_pdf_with_mineru(pdf_path, force_reparse=False, **kwargs):
        return (
            {
                "path": pdf_path,
                "doc_id": "doc1",
                "passages": [
                    {
                        "passage_id": "doc1:p1:c0",
                        "doc_id": "doc1",
                        "page": 1,
                        "chunk_idx": 0,
                        "block_type": "paragraph",
                        "anchors": [],
                        "text_for_embed": "A test passage",
                        "display_text": "[PDF_META doc=doc1 | page=1 | type=paragraph] A test passage",
                    }
                ],
            },
            [],
        )

    class FakeRAG:
        def index(self, passages, images, pdf_docs):
            self.indexed = (passages, images, pdf_docs)

        def qa(self, questions):
            q = questions[0]
            if q["question_id"] == "bad-q":
                raise RuntimeError("synthetic failure")
            return [
                {
                    "question": q["question"],
                    "gold_answer": q.get("answer", ""),
                    "target_doc_id": q["doc_id"],
                    "question_type": q.get("question_type", "general"),
                    "predicted_route": "general",
                    "used_route": "doc_aware_dense",
                    "selected_evidence_ids": ["passage::x"],
                    "scope_score_breakdown": {"passage::x": {"final": 1.0}},
                    "fallback_used": False,
                    "fallback_reason": "",
                    "pred_answer": "ok",
                }
            ]

    monkeypatch.setattr(run_mpdocvqa, "parse_pdf_with_mineru", fake_parse_pdf_with_mineru)
    monkeypatch.setattr(run_mpdocvqa, "build_rag_model", lambda args, dataset_name: FakeRAG())

    output_json = tmp_path / "results" / "predictions.json"
    rc = run_mpdocvqa.main(
        [
            "--mpdocvqa_root",
            str(root),
            "--split",
            "train",
            "--output_json",
            str(output_json),
        ]
    )
    assert rc == 0
    with open(output_json, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    assert len(predictions) == 2
    assert predictions[0]["selected_evidence_ids"] == ["passage::x"]
    assert predictions[1]["used_route"] == "error"
    assert predictions[1]["fallback_reason"] == "question_exception"
