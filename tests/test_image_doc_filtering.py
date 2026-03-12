from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doc_aware import resolve_allowed_doc_hash_ids


def test_resolve_allowed_doc_hash_ids_returns_empty_list_for_missing_doc_images():
    metadata = {
        "image-a": {"doc_id": "doc-a"},
        "image-b": {"doc_id": "doc-b"},
    }
    mapping = {
        "doc-a": ["image-a"],
        "doc-b": ["image-b"],
    }
    assert resolve_allowed_doc_hash_ids("doc-c", metadata, mapping) == []


def test_resolve_allowed_doc_hash_ids_uses_normalized_doc_id():
    metadata = {
        "image-a": {"doc_id": "SciAdv.ADR4808.PMC466939.pdf"},
    }
    mapping = {}
    assert resolve_allowed_doc_hash_ids("sciadv.adr4808.pmc466939", metadata, mapping) == ["image-a"]
