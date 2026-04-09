import re
from collections import defaultdict
from typing import Dict, List, Tuple

from src.doc_aware import detect_weak_parser_coverage, normalize_doc_id
from src.m3_lite import extract_figure_token_candidates


def _is_main_paper_token(token: str) -> bool:
    return not bool(re.search(r"\b(?:fig\.?|table)\s*S\d+\b", token or "", flags=re.IGNORECASE))


def _join_page_text(passage_rows, limit=3):
    text_bits = []
    for row in sorted(
        passage_rows,
        key=lambda x: (
            x.get("page") if isinstance(x.get("page"), int) else 10**9,
            int(x.get("chunk_idx", 10**6)),
        ),
    )[:limit]:
        text = (row.get("text_for_embed") or "").strip()
        if text:
            text_bits.append(text)
    return " ".join(text_bits)


def build_doc_context(passage_metadata_by_hash, image_metadata_by_hash, pdf_structure_state, target_doc_id):
    doc_id = normalize_doc_id(target_doc_id)
    page_nodes = pdf_structure_state.get("doc_to_page_nodes", {}).get(doc_id, [])
    weak_coverage, weak_reason = detect_weak_parser_coverage(
        doc_id,
        pdf_structure_state.get("doc_to_page_nodes", {}),
        build_doc_to_image_hash_ids(image_metadata_by_hash),
    )
    return {
        "doc_id": doc_id,
        "page_nodes": page_nodes,
        "page_count": len(page_nodes),
        "image_count": len(build_doc_to_image_hash_ids(image_metadata_by_hash).get(doc_id, [])),
        "weak_parser_coverage": weak_coverage,
        "weak_parser_reason": weak_reason,
    }


def build_doc_to_image_hash_ids(image_metadata_by_hash):
    out = defaultdict(list)
    for hid, meta in image_metadata_by_hash.items():
        doc_id = normalize_doc_id(meta.get("doc_id", ""))
        if doc_id:
            out[doc_id].append(hid)
    return dict(out)


def collect_plan_candidates_from_context(
    passage_metadata_by_hash,
    image_metadata_by_hash,
    pdf_structure_state,
    target_doc_id,
    evidence_plan,
    question_info,
):
    doc_id = normalize_doc_id(target_doc_id)
    if not doc_id:
        return [], build_doc_context(passage_metadata_by_hash, image_metadata_by_hash, pdf_structure_state, doc_id)

    doc_passages = []
    page_to_passages = defaultdict(list)
    for hid, meta in passage_metadata_by_hash.items():
        if normalize_doc_id(meta.get("doc_id", "")) != doc_id:
            continue
        row = dict(meta)
        row["hash_id"] = hid
        doc_passages.append(row)
        if isinstance(row.get("page"), int):
            page_to_passages[row["page"]].append(row)

    candidates = []
    for row in doc_passages:
        text = (row.get("text_for_embed") or "").strip()
        candidates.append(
            {
                "candidate_id": f"passage::{row['hash_id']}",
                "candidate_type": "passage",
                "doc_id": doc_id,
                "page": row.get("page"),
                "text": text,
                "block_type": row.get("block_type", ""),
                "anchors": list(row.get("anchors") or []),
                "is_main_paper": True,
                "has_image": (row.get("block_type") or "").lower() == "image",
                "metadata": {"hash_id": row["hash_id"], "chunk_idx": row.get("chunk_idx"), "display_text": row.get("display_text", text)},
                "source_node_ids": [row["hash_id"]],
            }
        )

    for page_node_id in pdf_structure_state.get("doc_to_page_nodes", {}).get(doc_id, []):
        page_meta = pdf_structure_state.get("page_nodes", {}).get(page_node_id, {})
        page_num = page_meta.get("page")
        page_rows = page_to_passages.get(page_num, [])
        candidates.append(
            {
                "candidate_id": page_node_id,
                "candidate_type": "page",
                "doc_id": doc_id,
                "page": page_num,
                "text": _join_page_text(page_rows),
                "block_type": "page",
                "anchors": [],
                "is_main_paper": True,
                "has_image": bool(page_meta.get("image_hash_ids")),
                "metadata": {
                    "page_node_id": page_node_id,
                    "is_front_matter": page_meta.get("is_front_matter", False),
                    "passage_hash_ids": list(page_meta.get("passage_hash_ids", [])),
                    "image_hash_ids": list(page_meta.get("image_hash_ids", [])),
                    "virtual_page": page_meta.get("virtual_page", False),
                },
                "source_node_ids": [page_node_id] + list(page_meta.get("passage_hash_ids", [])),
            }
        )

    for item in extract_figure_token_candidates(doc_passages, max_candidates=64):
        source_hash_ids = []
        for row in doc_passages:
            if row.get("page") == item.get("page") and row.get("chunk_idx") == item.get("chunk_idx"):
                source_hash_ids = [row["hash_id"]]
                break
        candidates.append(
            {
                "candidate_id": f"figure_token::{doc_id}::{item.get('page')}::{item.get('chunk_idx')}::{item.get('token')}",
                "candidate_type": "figure_token",
                "doc_id": doc_id,
                "page": item.get("page"),
                "text": item.get("token", ""),
                "block_type": "figure_token",
                "anchors": list(item.get("anchors") or []),
                "is_main_paper": _is_main_paper_token(item.get("token", "")),
                "has_image": False,
                "metadata": {
                    "token": item.get("token", ""),
                    "chunk_idx": item.get("chunk_idx"),
                    "supplementary": item.get("supplementary", False),
                    "snippet": item.get("snippet", ""),
                },
                "source_node_ids": source_hash_ids,
            }
        )

    for ihid, meta in image_metadata_by_hash.items():
        if normalize_doc_id(meta.get("doc_id", "")) != doc_id:
            continue
        candidates.append(
            {
                "candidate_id": f"image::{ihid}",
                "candidate_type": "image",
                "doc_id": doc_id,
                "page": meta.get("page"),
                "text": (meta.get("caption_hint") or meta.get("path") or "").strip(),
                "block_type": "image",
                "anchors": [],
                "is_main_paper": True,
                "has_image": True,
                "metadata": {"hash_id": ihid, "path": meta.get("path"), "caption_hint": meta.get("caption_hint", "")},
                "source_node_ids": [ihid],
            }
        )

    doc_context = build_doc_context(passage_metadata_by_hash, image_metadata_by_hash, pdf_structure_state, doc_id)
    return candidates, doc_context


def collect_plan_candidates(rag_model, target_doc_id, evidence_plan, question_info):
    return collect_plan_candidates_from_context(
        rag_model.passage_metadata_by_hash,
        rag_model.image_metadata_by_hash,
        rag_model.pdf_structure_state,
        target_doc_id,
        evidence_plan,
        question_info,
    )
