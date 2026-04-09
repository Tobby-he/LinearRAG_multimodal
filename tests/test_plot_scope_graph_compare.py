from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_scope_graph_compare as plotter


def test_build_baseline_graph_data():
    prediction = {
        "question": "[doc1] What is the name of the company?",
        "sorted_passage": [
            "[PDF_META doc=doc1 | page=0 | type=ocr_page] ITC LIMITED ANNUAL REPORT",
            "[PDF_META doc=doc1 | page=1 | type=ocr_page] COMPANY OVERVIEW",
        ],
        "pred_answer": "ITC Limited",
    }
    graph = plotter.build_baseline_graph_data(prediction, max_passages=2)
    assert len(graph["nodes"]) == 4
    assert ("q", "p1", "retrieve") in graph["edges"]
    assert any(node["type"] == "answer" for node in graph["nodes"])


def test_build_scope_graph_data_uses_real_node_types():
    prediction = {
        "question": "[doc1] What is the name of the company?",
        "doc_id": "doc1",
        "target_doc_id": "doc1",
        "predicted_route": "general",
        "pred_answer": "ITC Limited",
        "selected_evidence_pages": [0, 1],
    }
    structure_state = {}
    passage_metadata = {
        "passage-a": {
            "doc_id": "doc1",
            "page": 0,
            "chunk_idx": 0,
            "text_for_embed": "ITC LIMITED ANNUAL REPORT",
        },
        "passage-b": {
            "doc_id": "doc1",
            "page": 1,
            "chunk_idx": 0,
            "text_for_embed": "COMPANY OVERVIEW",
        },
    }
    image_metadata = {
        "image-a": {"doc_id": "doc1", "page": 0, "path": r"E:\images\doc1_p0.jpg"},
    }
    graph = plotter.build_scope_graph_data(
        prediction,
        structure_state,
        passage_metadata,
        image_metadata,
        max_passages=2,
        max_images=1,
        max_pages=2,
    )
    node_types = {node["type"] for node in graph["nodes"]}
    assert {"question", "plan", "page", "passage", "image", "answer"} <= node_types
    assert ("q", "plan", "route") in graph["edges"]
    assert ("page_0", "image_1", "visual") in graph["edges"]
