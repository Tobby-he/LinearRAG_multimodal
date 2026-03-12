import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_parser import _build_passage_records_from_content_data


def test_nested_content_list_propagates_page_and_anchor():
    data = [
        [
            {
                "type": "title",
                "content": {"title_content": [{"type": "text", "content": "Paper Title"}], "level": 1},
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Intro on first page."},
                    ]
                },
            },
        ],
        [
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "See Fig. 1 for the setup and Table 1 for parameters."},
                    ]
                },
            }
        ],
    ]
    records = _build_passage_records_from_content_data(data, "docx")
    assert records
    assert {r.page for r in records if r.page is not None} == {1, 2}
    page2 = [r for r in records if r.page == 2]
    assert page2
    assert any("ref:1" in r.anchors for r in page2)
    assert any("Fig. 1" in r.text_for_embed or "Table 1" in r.text_for_embed for r in page2)

