import glob
import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from src.doc_aware import (
    ImageRecord,
    PageRecord,
    PassageRecord,
    derive_doc_id_from_pdf_path,
)

PAGE_HEADER_RE = re.compile(r"^#\s*Page\s+(\d+)\s*$", re.IGNORECASE)
FIGURE_ANCHOR_RE = re.compile(r"\b(?:fig(?:ure)?\.?|table)\s*([0-9]+[A-Za-z]?)\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if chunk_size <= overlap:
        overlap = 0

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
    return chunks


def _run_mineru(pdf_path: str, output_dir: str) -> bool:
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["mineru", "-p", pdf_path, "-o", output_dir]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("mineru command not found. Please install MinerU in current environment.")
    except Exception as e:
        raise RuntimeError(f"failed to execute mineru: {e}")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(f"mineru parse failed for '{pdf_path}'. {detail}")
    return True


def _find_primary_md(parse_root: str) -> Optional[str]:
    # Prefer mineru standard output: */auto/*.md
    md_files = glob.glob(os.path.join(parse_root, "**", "auto", "*.md"), recursive=True)
    if not md_files:
        md_files = glob.glob(os.path.join(parse_root, "**", "*.md"), recursive=True)
    if not md_files:
        return None
    return sorted(md_files, key=lambda p: (len(p), p))[0]


def _find_content_json(md_path: str) -> Optional[str]:
    md_dir = os.path.dirname(md_path)
    stems = [
        os.path.splitext(os.path.basename(md_path))[0] + "_content_list_v2.json",
        os.path.splitext(os.path.basename(md_path))[0] + "_content_list.json",
        "content_list_v2.json",
        "content_list.json",
    ]
    for name in stems:
        candidate = os.path.join(md_dir, name)
        if os.path.exists(candidate):
            return candidate

    jsons = glob.glob(os.path.join(md_dir, "*_content_list_v2.json"))
    if jsons:
        return sorted(jsons)[0]
    jsons = glob.glob(os.path.join(md_dir, "*_content_list.json"))
    if jsons:
        return sorted(jsons)[0]
    return None


def _extract_page_from_item(item: Dict) -> Optional[int]:
    for key in ("page_no", "page_id", "page", "page_idx"):
        if key in item and item[key] is not None:
            try:
                val = int(item[key])
                # mineru often uses 0-based page idx
                return val + 1 if val >= 0 else None
            except Exception:
                pass

    # bbox may carry page index in some formats
    bbox = item.get("bbox")
    if isinstance(bbox, dict):
        p = bbox.get("page")
        if p is not None:
            try:
                val = int(p)
                return val + 1 if val >= 0 else None
            except Exception:
                return None
    return None


def _collect_text_fragments(obj) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj)
        return out
    if isinstance(obj, list):
        for item in obj:
            out.extend(_collect_text_fragments(item))
        return out
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in {"path", "bbox", "html"}:
                continue
            out.extend(_collect_text_fragments(val))
    return out


def _extract_text_from_item(item: Dict) -> str:
    return normalize_text(" ".join(_collect_text_fragments(item)))


def _extract_type_from_item(item: Dict) -> str:
    for key in ("type", "category", "block_type", "text_type"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.lower()

    block = item.get("block")
    if isinstance(block, dict):
        val = block.get("type")
        if isinstance(val, str) and val.strip():
            return val.lower()

    return "text"


def _iter_content_items_with_pages(data) -> List[Tuple[Optional[int], Dict]]:
    rows: List[Tuple[Optional[int], Dict]] = []
    if not isinstance(data, list):
        return rows
    for outer_idx, outer_item in enumerate(data):
        if isinstance(outer_item, list):
            default_page = outer_idx + 1
            for inner in outer_item:
                if isinstance(inner, dict):
                    rows.append((_extract_page_from_item(inner) or default_page, inner))
        elif isinstance(outer_item, dict):
            rows.append((_extract_page_from_item(outer_item), outer_item))
    return rows


def _build_passage_records_from_content_data(data, doc_id: str) -> List[PassageRecord]:
    if not isinstance(data, list):
        return []

    flat_items = _iter_content_items_with_pages(data)
    if not flat_items:
        return []

    # First pass: collect nearby figure/table anchors per page.
    page_anchors: Dict[int, List[str]] = {}
    rows: List[Tuple[Optional[int], str, str]] = []

    for default_page, item in flat_items:
        text = normalize_text(_extract_text_from_item(item))
        if not text:
            continue
        page = _extract_page_from_item(item) or default_page
        item_type = _extract_type_from_item(item)
        rows.append((page, item_type, text))

        if page is not None:
            anchors = FIGURE_ANCHOR_RE.findall(text)
            if anchors:
                labels = [f"ref:{a}" for a in anchors]
                page_anchors.setdefault(page, []).extend(labels)

    for page, anchors in list(page_anchors.items()):
        # Deduplicate while preserving order.
        seen = set()
        dedup = []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                dedup.append(a)
        page_anchors[page] = dedup[:8]

    records: List[PassageRecord] = []
    chunk_idx = 0
    for page, item_type, text in rows:
        base = text
        if len(base) > 1500:
            piece_list = split_text(base, chunk_size=1200, overlap=150)
        else:
            piece_list = [base]

        for piece in piece_list:
            # Cross-page hint: include neighboring pages' anchors in metadata only.
            anchors: List[str] = []
            if page is not None:
                local = page_anchors.get(page, [])[:3]
                prev_ = page_anchors.get(page - 1, [])[:2]
                next_ = page_anchors.get(page + 1, [])[:2]
                anchors = local + prev_ + next_
            passage_id = f"{doc_id}:p{page if page is not None else 'na'}:c{chunk_idx}"
            display_prefix = f"[PDF_META doc={doc_id}"
            if page is not None:
                display_prefix += f" | page={page}"
            if item_type:
                display_prefix += f" | type={item_type}"
            if anchors:
                display_prefix += f" | anchors={','.join(anchors)}"
            display_prefix += "] "
            records.append(
                PassageRecord(
                    passage_id=passage_id,
                    doc_id=doc_id,
                    page=page,
                    chunk_idx=chunk_idx,
                    block_type=item_type or "text",
                    anchors=anchors,
                    text_for_embed=piece,
                    display_text=display_prefix + piece,
                )
            )
            chunk_idx += 1

    return records


def _build_passage_records_from_json(content_json_path: str, doc_id: str) -> List[PassageRecord]:
    try:
        with open(content_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return _build_passage_records_from_content_data(data, doc_id)


def _build_passage_records_from_md(md_text: str, doc_id: str) -> List[PassageRecord]:
    lines = (md_text or "").splitlines()
    records: List[PassageRecord] = []
    page = None
    buffer: List[str] = []
    chunk_idx = 0

    def flush_buffer(current_page: Optional[int]) -> None:
        nonlocal chunk_idx
        if not buffer:
            return
        paragraph = normalize_text("\n".join(buffer))
        buffer.clear()
        if not paragraph:
            return
        parts = split_text(paragraph, chunk_size=1200, overlap=150) or [paragraph]
        for p in parts:
            anchors = [f"ref:{a}" for a in FIGURE_ANCHOR_RE.findall(p)][:5]
            passage_id = f"{doc_id}:p{current_page if current_page is not None else 'na'}:c{chunk_idx}"
            prefix = f"[PDF_META doc={doc_id}"
            if current_page is not None:
                prefix += f" | page={current_page}"
            prefix += " | type=md"
            if anchors:
                prefix += f" | anchors={','.join(anchors)}"
            prefix += "] "
            records.append(
                PassageRecord(
                    passage_id=passage_id,
                    doc_id=doc_id,
                    page=current_page,
                    chunk_idx=chunk_idx,
                    block_type="md",
                    anchors=anchors,
                    text_for_embed=p,
                    display_text=prefix + p,
                )
            )
            chunk_idx += 1

    for line in lines:
        m = PAGE_HEADER_RE.match(line.strip())
        if m:
            flush_buffer(page)
            page = int(m.group(1))
            continue

        if line.strip() == "":
            flush_buffer(page)
        else:
            buffer.append(line)

    flush_buffer(page)
    return records


def _build_image_records_from_json(
    content_json_path: str,
    doc_id: str,
    parse_out_dir: str,
    image_out_dir: str,
    source_pdf: str,
) -> List[Dict]:
    try:
        with open(content_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    image_records: List[Dict] = []
    copied = set()
    image_idx = 0
    for default_page, item in _iter_content_items_with_pages(data):
        if not isinstance(item, dict):
            continue
        item_type = _extract_type_from_item(item)
        if item_type not in {"image", "table"}:
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        image_source = content.get("image_source") if isinstance(content, dict) else {}
        src_rel = image_source.get("path") if isinstance(image_source, dict) else None
        if not src_rel:
            continue
        img_src = os.path.join(os.path.dirname(content_json_path), src_rel)
        if not os.path.exists(img_src):
            img_src = os.path.join(parse_out_dir, src_rel)
        if not os.path.exists(img_src):
            continue
        img_name = os.path.basename(img_src)
        img_dst = os.path.join(image_out_dir, img_name)
        key = os.path.normcase(img_dst)
        if key in copied:
            continue
        copied.add(key)
        try:
            shutil.copy2(img_src, img_dst)
        except Exception:
            continue
        image_id = f"{doc_id}:img{image_idx}"
        image_idx += 1
        image_records.append(
            ImageRecord(
                image_id=image_id,
                doc_id=doc_id,
                page=_extract_page_from_item(item) or default_page,
                path=img_dst,
                caption_hint=_extract_text_from_item(item) or None,
                source_pdf=source_pdf,
            ).to_dict()
        )
    return image_records


def parse_pdf_with_mineru(
    pdf_path: str,
    parse_base_dir: str = "import/pdf_parsed",
    image_base_dir: str = "import/pdf_images",
    force_reparse: bool = False,
) -> Tuple[Dict, List[Dict]]:
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_id = derive_doc_id_from_pdf_path(pdf_path)
    pdf_base = os.path.basename(pdf_path)
    parse_out_dir = os.path.join(parse_base_dir, pdf_name)
    image_out_dir = os.path.join(image_base_dir, pdf_base)

    os.makedirs(parse_out_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)

    md_path = _find_primary_md(parse_out_dir)
    content_json_path = _find_content_json(md_path) if md_path else None
    has_cached_parse = bool(
        md_path
        and os.path.exists(md_path)
        and content_json_path
        and os.path.exists(content_json_path)
    )

    if force_reparse or not has_cached_parse:
        _run_mineru(pdf_path, parse_out_dir)
        md_path = _find_primary_md(parse_out_dir)
        content_json_path = _find_content_json(md_path) if md_path else None

    if md_path and os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
            md_text = f.read()
    else:
        md_text = ""

    if content_json_path and os.path.exists(content_json_path):
        passage_records = _build_passage_records_from_json(content_json_path, doc_id=doc_id)
    else:
        passage_records = _build_passage_records_from_md(md_text, doc_id=doc_id)

    if not passage_records:
        fallback_parts = split_text(md_text, chunk_size=1200, overlap=150)
        passage_records = []
        for i, part in enumerate(fallback_parts):
            pid = f"{doc_id}:pna:c{i}"
            passage_records.append(
                PassageRecord(
                    passage_id=pid,
                    doc_id=doc_id,
                    page=None,
                    chunk_idx=i,
                    block_type="text",
                    anchors=[],
                    text_for_embed=part,
                    display_text=f"[PDF_META doc={doc_id} | type=text] {part}",
                )
            )

    image_records: List[Dict] = []
    if content_json_path and os.path.exists(content_json_path):
        image_records = _build_image_records_from_json(
            content_json_path=content_json_path,
            doc_id=doc_id,
            parse_out_dir=parse_out_dir,
            image_out_dir=image_out_dir,
            source_pdf=pdf_path,
        )
    if not image_records:
        copied = set()
        image_idx = 0
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img_src in glob.glob(os.path.join(parse_out_dir, "**", ext), recursive=True):
                img_name = os.path.basename(img_src)
                img_dst = os.path.join(image_out_dir, img_name)
                key = os.path.normcase(img_dst)
                if key in copied:
                    continue
                copied.add(key)
                try:
                    shutil.copy2(img_src, img_dst)
                except Exception:
                    continue
                image_id = f"{doc_id}:img{image_idx}"
                image_idx += 1
                page_num = None
                page_match = re.search(r"(?:page|p)[_\\-]?(\\d{1,4})", img_name, flags=re.IGNORECASE)
                if page_match:
                    try:
                        page_num = int(page_match.group(1))
                    except Exception:
                        page_num = None
                image_records.append(
                    ImageRecord(
                        image_id=image_id,
                        doc_id=doc_id,
                        page=page_num,
                        path=img_dst,
                        caption_hint=None,
                        source_pdf=pdf_path,
                    ).to_dict()
                )

    page_to_passages: Dict[int, List[str]] = {}
    for rec in passage_records:
        if rec.page is None:
            continue
        page_to_passages.setdefault(rec.page, []).append(rec.passage_id)
    page_to_images: Dict[int, List[str]] = {}
    for img in image_records:
        if img.get("page") is None:
            continue
        page_to_images.setdefault(int(img["page"]), []).append(img["image_id"])
    page_records = [
        PageRecord(
            page_id=f"{doc_id}:page{p}",
            doc_id=doc_id,
            page=p,
            passage_ids=page_to_passages.get(p, []),
            image_ids=page_to_images.get(p, []),
        ).to_dict()
        for p in sorted(set(page_to_passages.keys()) | set(page_to_images.keys()))
    ]

    doc = {
        "path": pdf_path,
        "doc_id": doc_id,
        "content": md_text,
        "passages": [r.to_dict() for r in passage_records],
        "pages": page_records,
        # Backward-compatible field
        "xref_chunks": [r.display_text for r in passage_records],
        "parse_dir": parse_out_dir,
    }
    return doc, image_records


def load_pdf_documents(pdf_paths_file: str, force_reparse: bool = False) -> Tuple[List[Dict], List[Dict]]:
    with open(pdf_paths_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "pdf_paths" in data and isinstance(data["pdf_paths"], list):
            pdf_paths = data["pdf_paths"]
        else:
            pdf_paths = [v for v in data.values() if isinstance(v, str)]
    elif isinstance(data, list):
        pdf_paths = data
    else:
        pdf_paths = []

    docs: List[Dict] = []
    images: List[Dict] = []

    for item in pdf_paths:
        if isinstance(item, str):
            pdf_path = item
        elif isinstance(item, dict):
            pdf_path = item.get("path") or item.get("pdf_path")
        else:
            pdf_path = None

        if not pdf_path:
            continue

        norm = os.path.normpath(pdf_path)
        if not os.path.exists(norm):
            continue

        doc, image_records = parse_pdf_with_mineru(norm, force_reparse=force_reparse)
        docs.append(doc)
        images.extend(image_records)

    return docs, images
