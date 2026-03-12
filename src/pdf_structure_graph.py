import re
from collections import defaultdict

from src.doc_aware import normalize_doc_id


def infer_page_from_image_path(path):
    if not path:
        return None
    name = str(path).replace("\\", "/").split("/")[-1]
    m = re.search(r"(?:page|p)[_\-]?(\d{1,4})", name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _normalize_anchor(anchor):
    a = (anchor or "").strip().lower()
    if not a:
        return ""
    a = a.replace(" ", "")
    if a.startswith("ref:"):
        a = a[4:]
    return a


def _page_node_id(doc_id, page):
    return f"page::{normalize_doc_id(doc_id)}::{page}"


def _visual_node_id(doc_id, anchor):
    return f"visual::{normalize_doc_id(doc_id)}::{anchor}"


def build_pdf_structure_state(pdf_data, passage_metadata_by_hash, image_metadata_by_hash, infer_page_fn=None):
    passage_id_to_hash = {}
    doc_to_passages = defaultdict(list)
    for hid, meta in passage_metadata_by_hash.items():
        pid = meta.get("passage_id")
        if pid:
            passage_id_to_hash[pid] = hid
        doc_id = normalize_doc_id(meta.get("doc_id") or "")
        if doc_id:
            doc_to_passages[doc_id].append((hid, meta))

    page_nodes = {}
    visual_nodes = {}
    doc_to_page_nodes = defaultdict(list)
    edges = []
    passage_adjacent_map = defaultdict(set)

    for doc in pdf_data or []:
        doc_id = normalize_doc_id(doc.get("doc_id") or "")
        if not doc_id:
            continue
        pages = doc.get("pages") or []
        if pages:
            pages = sorted(pages, key=lambda x: int(x.get("page", 10**9)))
            prev_node = None
            for page_rec in pages:
                page_num = page_rec.get("page")
                if not isinstance(page_num, int):
                    continue
                page_node = _page_node_id(doc_id, page_num)
                page_nodes[page_node] = {
                    "page_node_id": page_node,
                    "doc_id": doc_id,
                    "page": page_num,
                    "is_front_matter": page_num <= 2,
                    "virtual_page": False,
                    "passage_hash_ids": [],
                    "image_hash_ids": [],
                }
                doc_to_page_nodes[doc_id].append(page_node)

                for pid in page_rec.get("passage_ids", []):
                    phid = passage_id_to_hash.get(pid)
                    if not phid:
                        continue
                    page_nodes[page_node]["passage_hash_ids"].append(phid)
                    edges.append((phid, page_node, "passage_page", 1.0))

                if prev_node:
                    edges.append((prev_node, page_node, "page_adjacent", 0.7))
                prev_node = page_node

    # Fallback page nodes from passage metadata when parser page records are missing.
    for doc_id, pitems in doc_to_passages.items():
        if doc_to_page_nodes.get(doc_id):
            continue
        page_nums = sorted({m.get("page") for _, m in pitems if isinstance(m.get("page"), int)})
        if not page_nums and pitems:
            # Legacy fallback: keep a single virtual page so structure lookup remains usable.
            page_nums = [1]
        prev_node = None
        for pnum in page_nums:
            page_node = _page_node_id(doc_id, pnum)
            page_nodes[page_node] = {
                "page_node_id": page_node,
                "doc_id": doc_id,
                "page": pnum,
                "is_front_matter": pnum <= 2,
                "virtual_page": not any(isinstance(m.get("page"), int) for _, m in pitems),
                "passage_hash_ids": [],
                "image_hash_ids": [],
            }
            doc_to_page_nodes[doc_id].append(page_node)
            for phid, meta in pitems:
                if (meta.get("page") == pnum) or (not isinstance(meta.get("page"), int) and pnum == 1):
                    page_nodes[page_node]["passage_hash_ids"].append(phid)
                    edges.append((phid, page_node, "passage_page", 1.0))
            if prev_node:
                edges.append((prev_node, page_node, "page_adjacent", 0.7))
            prev_node = page_node

    # Passage adjacency from metadata order (page + chunk_idx).
    for doc_id, pitems in doc_to_passages.items():
        ordered = []
        for phid, meta in pitems:
            page = meta.get("page")
            chunk_idx = int(meta.get("chunk_idx", 0))
            page_order = page if isinstance(page, int) else 10**9
            ordered.append((page_order, chunk_idx, phid))
        ordered.sort(key=lambda x: (x[0], x[1]))
        for i in range(len(ordered) - 1):
            u = ordered[i][2]
            v = ordered[i + 1][2]
            edges.append((u, v, "passage_adjacent", 0.35))
            passage_adjacent_map[u].add(v)
            passage_adjacent_map[v].add(u)

    # Link images to pages when page is available or inferable.
    for ihid, meta in image_metadata_by_hash.items():
        doc_id = normalize_doc_id(meta.get("doc_id") or "")
        if not doc_id:
            continue
        page = meta.get("page")
        if not isinstance(page, int) and infer_page_fn:
            page = infer_page_fn(meta.get("path"))
        if not isinstance(page, int):
            continue
        page_node = _page_node_id(doc_id, page)
        if page_node not in page_nodes:
            page_nodes[page_node] = {
                "page_node_id": page_node,
                "doc_id": doc_id,
                "page": page,
                "is_front_matter": page <= 2,
                "virtual_page": False,
                "passage_hash_ids": [],
                "image_hash_ids": [],
            }
            doc_to_page_nodes[doc_id].append(page_node)
        page_nodes[page_node]["image_hash_ids"].append(ihid)
        edges.append((ihid, page_node, "image_page", 0.9))

    # Passage anchors -> visual nodes and visual -> page links.
    for phid, meta in passage_metadata_by_hash.items():
        doc_id = normalize_doc_id(meta.get("doc_id") or "")
        if not doc_id:
            continue
        page = meta.get("page")
        anchors = meta.get("anchors") or []
        for anchor in anchors:
            normalized = _normalize_anchor(anchor)
            if not normalized:
                continue
            vnode = _visual_node_id(doc_id, normalized)
            visual_nodes[vnode] = {
                "visual_node_id": vnode,
                "doc_id": doc_id,
                "anchor": normalized,
            }
            edges.append((phid, vnode, "passage_visual", 0.8))
            if isinstance(page, int):
                page_node = _page_node_id(doc_id, page)
                if page_node in page_nodes:
                    edges.append((vnode, page_node, "visual_page", 0.6))

    # Keep doc page ordering stable for debugging.
    for doc_id in list(doc_to_page_nodes.keys()):
        doc_to_page_nodes[doc_id] = sorted(
            set(doc_to_page_nodes[doc_id]),
            key=lambda x: int(str(x).rsplit("::", 1)[-1]) if str(x).rsplit("::", 1)[-1].isdigit() else 10**9,
        )

    return {
        "page_nodes": page_nodes,
        "visual_nodes": visual_nodes,
        "doc_to_page_nodes": dict(doc_to_page_nodes),
        "edges": edges,
        "passage_adjacent_map": passage_adjacent_map,
    }
