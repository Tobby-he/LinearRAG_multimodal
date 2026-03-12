from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pdf_structure_graph import build_pdf_structure_state


def _mock_state():
    passage_metadata_by_hash = {
        "ph1": {
            "passage_id": "doca:p1:c0",
            "doc_id": "doca",
            "page": 1,
            "chunk_idx": 0,
            "anchors": ["ref:1"],
        },
        "ph2": {
            "passage_id": "doca:p1:c1",
            "doc_id": "doca",
            "page": 1,
            "chunk_idx": 1,
            "anchors": [],
        },
        "ph3": {
            "passage_id": "doca:p2:c2",
            "doc_id": "doca",
            "page": 2,
            "chunk_idx": 2,
            "anchors": ["ref:table1"],
        },
    }
    image_metadata_by_hash = {
        "ih1": {"image_id": "doca:img0", "doc_id": "doca", "page": 2, "path": "x.png"},
    }
    pdf_data = [
        {
            "doc_id": "doca",
            "pages": [
                {"page_id": "doca:page1", "doc_id": "doca", "page": 1, "passage_ids": ["doca:p1:c0", "doca:p1:c1"], "image_ids": []},
                {"page_id": "doca:page2", "doc_id": "doca", "page": 2, "passage_ids": ["doca:p2:c2"], "image_ids": []},
            ],
        }
    ]
    return build_pdf_structure_state(
        pdf_data=pdf_data,
        passage_metadata_by_hash=passage_metadata_by_hash,
        image_metadata_by_hash=image_metadata_by_hash,
    )


def test_page_nodes_created():
    s = _mock_state()
    assert "page::doca::1" in s["page_nodes"]
    assert "page::doca::2" in s["page_nodes"]


def test_passage_to_page_links_exist():
    s = _mock_state()
    edge_set = {(u, v, t) for u, v, t, _ in s["edges"]}
    assert ("ph1", "page::doca::1", "passage_page") in edge_set
    assert ("ph3", "page::doca::2", "passage_page") in edge_set


def test_page_adjacency_exists():
    s = _mock_state()
    edge_set = {(u, v, t) for u, v, t, _ in s["edges"]}
    assert ("page::doca::1", "page::doca::2", "page_adjacent") in edge_set


def test_adjacent_passage_links_exist_by_metadata_order():
    s = _mock_state()
    edge_set = {(u, v, t) for u, v, t, _ in s["edges"]}
    assert ("ph1", "ph2", "passage_adjacent") in edge_set
    assert ("ph2", "ph3", "passage_adjacent") in edge_set


def test_anchor_creates_discoverable_visual_connection():
    s = _mock_state()
    # visual node derived from anchor "ref:1"
    assert "visual::doca::1" in s["visual_nodes"]
    edge_set = {(u, v, t) for u, v, t, _ in s["edges"]}
    assert ("ph1", "visual::doca::1", "passage_visual") in edge_set


def test_doc_id_normalization_in_structure_state_keys():
    passage_metadata_by_hash = {
        "p1": {"passage_id": "X:p1:c0", "doc_id": "SciAdv.Adp1439.PMC466956.PDF", "page": 1, "chunk_idx": 0, "anchors": []}
    }
    pdf_data = [
        {
            "doc_id": "SCIADV.ADP1439.PMC466956",
            "pages": [
                {
                    "page_id": "x:page1",
                    "doc_id": "SCIADV.ADP1439.PMC466956",
                    "page": 1,
                    "passage_ids": ["X:p1:c0"],
                    "image_ids": [],
                }
            ],
        }
    ]
    s = build_pdf_structure_state(pdf_data, passage_metadata_by_hash, image_metadata_by_hash={})
    assert "sciadv.adp1439.pmc466956" in s["doc_to_page_nodes"]
    assert "page::sciadv.adp1439.pmc466956::1" in s["page_nodes"]


def test_virtual_page_fallback_when_page_metadata_missing():
    passage_metadata_by_hash = {
        "p1": {"passage_id": "docz:pna:c0", "doc_id": "docz", "page": None, "chunk_idx": 0, "anchors": []},
        "p2": {"passage_id": "docz:pna:c1", "doc_id": "docz", "page": None, "chunk_idx": 1, "anchors": []},
    }
    s = build_pdf_structure_state(pdf_data=[], passage_metadata_by_hash=passage_metadata_by_hash, image_metadata_by_hash={})
    assert "docz" in s["doc_to_page_nodes"]
    assert "page::docz::1" in s["page_nodes"]
    edge_set = {(u, v, t) for u, v, t, _ in s["edges"]}
    assert ("p1", "page::docz::1", "passage_page") in edge_set
    assert ("p2", "page::docz::1", "passage_page") in edge_set
