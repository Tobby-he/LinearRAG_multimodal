from dataclasses import asdict, dataclass
import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


QUESTION_TYPE_TITLE = "title"
QUESTION_TYPE_FIRST_FIGURE = "first_figure_token"
QUESTION_TYPE_SUMMARY = "summary"
QUESTION_TYPE_QUANT_PLUS_FIGURE = "quant_plus_first_figure"
QUESTION_TYPE_GENERAL = "general"
QUESTION_TYPE_CHART_NUMERIC = "chart_numeric"
SUPPORTED_QUESTION_TYPES = {
    QUESTION_TYPE_TITLE,
    QUESTION_TYPE_FIRST_FIGURE,
    QUESTION_TYPE_SUMMARY,
    QUESTION_TYPE_QUANT_PLUS_FIGURE,
    QUESTION_TYPE_GENERAL,
    QUESTION_TYPE_CHART_NUMERIC,
}


@dataclass
class PassageRecord:
    passage_id: str
    doc_id: str
    page: Optional[int]
    chunk_idx: int
    block_type: str
    anchors: List[str]
    text_for_embed: str
    display_text: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ImageRecord:
    image_id: str
    doc_id: str
    page: Optional[int]
    path: str
    caption_hint: Optional[str]
    source_pdf: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PageRecord:
    page_id: str
    doc_id: str
    page: int
    passage_ids: List[str]
    image_ids: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def derive_doc_id_from_pdf_path(pdf_path: str) -> str:
    return normalize_doc_id(pdf_path)


def normalize_doc_id(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # Strip leading [doc] wrapper if accidentally provided.
    if s.startswith("[") and s.endswith("]") and len(s) > 2:
        s = s[1:-1].strip()
    s = s.replace("\\", "/").split("/")[-1]
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    return s.strip().lower()


def parse_doc_id_from_question(question: str) -> Tuple[str, str]:
    if not isinstance(question, str):
        return "", question
    m = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", question)
    if not m:
        return "", question.strip()
    return normalize_doc_id(m.group(1)), m.group(2).strip()


def infer_question_type(question: str) -> str:
    q = (question or "").lower()
    if (
        "exact full paper title" in q
        or "exact title" in q
        or "official title" in q
        or "complete paper title" in q
        or "complete article title" in q
        or "full article title" in q
    ):
        return QUESTION_TYPE_TITLE
    if "summarize the main contribution" in q:
        return QUESTION_TYPE_SUMMARY
    if re.search(r"\b(?:according to|from)\s+(?:fig(?:ure)?\.?|table)\s*[s]?\d+[a-z]?\b", q) and re.search(r"\b(?:how many|what .*?(?:value|values|number|numbers|fwhm|power|powers|studies))\b", q):
        return QUESTION_TYPE_CHART_NUMERIC
    if "first figure/table caption token" in q and "combine" not in q:
        return QUESTION_TYPE_FIRST_FIGURE
    if ("combine one key quantitative statement" in q and "first figure/table" in q) or ("numeric result" in q and "first main-paper figure/table" in q):
        return QUESTION_TYPE_QUANT_PLUS_FIGURE
    return QUESTION_TYPE_GENERAL


def resolve_question_type(question_info, raw_question: str, logger: Optional[logging.Logger] = None) -> str:
    inferred = infer_question_type(raw_question)
    if not isinstance(question_info, dict):
        return inferred

    dataset_question_type = question_info.get("question_type")
    if dataset_question_type is None:
        return inferred

    normalized = str(dataset_question_type).strip()
    if not normalized:
        if logger is not None:
            logger.info(
                "question_type fallback reason=missing_or_empty raw_question=%s inferred_question_type=%s",
                raw_question,
                inferred,
            )
        return inferred

    if normalized not in SUPPORTED_QUESTION_TYPES:
        if logger is not None:
            logger.warning(
                "question_type fallback reason=invalid dataset_question_type=%s raw_question=%s inferred_question_type=%s",
                normalized,
                raw_question,
                inferred,
            )
        return inferred

    return normalized


def filter_ids_by_doc_id(
    hash_ids: Sequence[str],
    metadata_by_hash: Dict[str, Dict],
    doc_id: str,
) -> List[str]:
    norm_doc_id = normalize_doc_id(doc_id)
    if not norm_doc_id:
        return list(hash_ids)
    out = []
    for hid in hash_ids:
        meta = metadata_by_hash.get(hid, {})
        if normalize_doc_id(meta.get("doc_id", "")) == norm_doc_id:
            out.append(hid)
    return out


def build_doc_to_hash_ids(metadata_by_hash: Dict[str, Dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for hid, meta in metadata_by_hash.items():
        doc_id = normalize_doc_id(meta.get("doc_id", ""))
        if not doc_id:
            continue
        out.setdefault(doc_id, []).append(hid)
    return out


def resolve_allowed_doc_hash_ids(
    target_doc_id: str,
    metadata_by_hash: Dict[str, Dict],
    doc_to_hash_ids: Dict[str, List[str]],
) -> Optional[List[str]]:
    norm_doc_id = normalize_doc_id(target_doc_id)
    if not norm_doc_id:
        return None
    allowed = doc_to_hash_ids.get(norm_doc_id)
    if allowed is not None:
        return allowed
    return filter_ids_by_doc_id(list(metadata_by_hash.keys()), metadata_by_hash, norm_doc_id)


def detect_weak_parser_coverage(doc_id: str, doc_to_page_nodes: Dict[str, List[str]], doc_to_image_hash_ids: Dict[str, List[str]]) -> Tuple[bool, str]:
    norm_doc_id = normalize_doc_id(doc_id)
    if not norm_doc_id:
        return False, ""
    num_pages = len(doc_to_page_nodes.get(norm_doc_id, []))
    num_images = len(doc_to_image_hash_ids.get(norm_doc_id, []))
    if num_pages <= 1 and num_images == 0:
        return True, f"weak parser coverage: pages={num_pages}, images={num_images}"
    return False, ""


def is_clean_text_for_embed(text: str) -> bool:
    # Embedding text should not carry metadata prefixes.
    return not text.strip().startswith("[PDF_META")
