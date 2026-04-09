from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_real_doc_structure as plotter


def test_collect_doc_graph():
    structure_state = {
        "page_nodes": {
            "page::doc1::0": {
                "page_node_id": "page::doc1::0",
                "doc_id": "doc1",
                "page": 0,
                "passage_hash_ids": ["p1", "p2"],
                "image_hash_ids": ["i1"],
            },
            "page::doc1::1": {
                "page_node_id": "page::doc1::1",
                "doc_id": "doc1",
                "page": 1,
                "passage_hash_ids": ["p3"],
                "image_hash_ids": [],
            },
        }
    }
    passage_metadata = {
        "p1": {"doc_id": "doc1", "page": 0},
        "p2": {"doc_id": "doc1", "page": 0},
        "p3": {"doc_id": "doc1", "page": 1},
    }
    image_metadata = {
        "i1": {"doc_id": "doc1", "page": 0, "path": "doc1_p0.jpg"},
    }
    pages, passages, images = plotter.collect_doc_graph(structure_state, passage_metadata, image_metadata, "doc1")
    assert len(pages) == 2
    assert len(passages) == 3
    assert len(images) == 1

