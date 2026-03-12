from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doc_aware import detect_weak_parser_coverage


def test_detect_weak_parser_coverage_for_single_page_zero_image_doc():
    weak, reason = detect_weak_parser_coverage(
        "sciadv.adr4808.pmc466939",
        {"sciadv.adr4808.pmc466939": ["page::sciadv.adr4808.pmc466939::1"]},
        {},
    )
    assert weak is True
    assert "pages=1" in reason
    assert "images=0" in reason


def test_detect_weak_parser_coverage_not_triggered_for_multi_page_doc():
    weak, reason = detect_weak_parser_coverage(
        "doc-a",
        {"doc-a": ["page::doc-a::1", "page::doc-a::2"]},
        {"doc-a": ["image-1"]},
    )
    assert weak is False
    assert reason == ""
