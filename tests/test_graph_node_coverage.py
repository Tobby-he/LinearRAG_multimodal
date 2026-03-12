from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_structure_graph import build_pdf_structure_state


def test_structure_state_contains_image_page_edges_when_image_pages_exist():
    passage_metadata = {
        "p1": {"passage_id": "doca:p1:c0", "doc_id": "doca", "page": 1, "chunk_idx": 0, "anchors": []},
    }
    image_metadata = {
        "img1": {"image_id": "doca:img0", "doc_id": "doca", "page": 1, "path": "x.png"},
    }
    state = build_pdf_structure_state(
        pdf_data=[{"doc_id": "doca", "pages": [{"page_id": "doca:page1", "doc_id": "doca", "page": 1, "passage_ids": ["doca:p1:c0"], "image_ids": ["doca:img0"]}]}],
        passage_metadata_by_hash=passage_metadata,
        image_metadata_by_hash=image_metadata,
    )
    edge_set = {(u, v, t) for u, v, t, _ in state["edges"]}
    assert ("img1", "page::doca::1", "image_page") in edge_set

