import re

from src.doc_aware import normalize_doc_id
from src.semantic_router import (
    GENERAL_ROUTE,
    FIGURE_ROUTE,
    QUANT_ROUTE,
    SUMMARY_ROUTE,
    TITLE_ROUTE,
)


def route_question_rule(question: str) -> str:
    q = (question or "").lower()
    if "exact full paper title" in q or "exact title" in q:
        return TITLE_ROUTE
    if "combine one key quantitative statement" in q and "first figure/table" in q:
        return QUANT_ROUTE
    if "first figure/table caption token" in q:
        return FIGURE_ROUTE
    if "summarize the main contribution" in q or "main contribution in one sentence" in q:
        return SUMMARY_ROUTE
    return GENERAL_ROUTE

FIG_TOKEN_RE = re.compile(r"\b(Fig(?:ure)?\.?\s*(?:S)?\d+[A-Za-z]?|Table\s*(?:S)?\d+[A-Za-z]?)\b", re.IGNORECASE)
LABEL_PREFIX_RE = re.compile(r"^(?:(?:title|paragraph|header|footer|caption|figure|table)\s+text|(?:title|paragraph|header|footer|caption|figure|table)|text)\s+", re.IGNORECASE)
QUANT_MARKER_RE = re.compile(
    r"("
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|percent|fold|times|day|days|hour|hours|min|minutes|s|ms|nm|μm|um|mW|W|bar|kDa|Da|tons?|samples?|patients?)"
    r"|turnover number"
    r"|l/b ratio"
    r"|accuracy"
    r"|yield"
    r"|increased? by"
    r"|decreased? by"
    r"|up to\s+\d"
    r")",
    re.IGNORECASE,
)


def route_question_hybrid(question: str, semantic_router=None, semantic_threshold=0.42):
    rule_route = route_question_rule(question)
    if rule_route != GENERAL_ROUTE:
        return {
            "predicted_route": rule_route,
            "route_source": "rule",
            "route_confidence": 1.0,
            "top_route_scores": {rule_route: 1.0},
            "matched_prototypes": [],
        }
    if semantic_router is None:
        return {
            "predicted_route": GENERAL_ROUTE,
            "route_source": "fallback",
            "route_confidence": 0.0,
            "top_route_scores": {GENERAL_ROUTE: 0.0},
            "matched_prototypes": [],
        }
    semantic_result = semantic_router.predict(question)
    if semantic_result["route_confidence"] < semantic_threshold:
        semantic_result = dict(semantic_result)
        semantic_result["predicted_route"] = GENERAL_ROUTE
        semantic_result["route_source"] = "fallback"
        return semantic_result
    semantic_result = dict(semantic_result)
    semantic_result["route_source"] = "semantic"
    return semantic_result


def route_question(question: str, semantic_router=None, semantic_threshold=0.42) -> str:
    return route_question_hybrid(
        question,
        semantic_router=semantic_router,
        semantic_threshold=semantic_threshold,
    )["predicted_route"]


def _clean_text(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    while True:
        cleaned = LABEL_PREFIX_RE.sub("", s).strip()
        if cleaned == s:
            break
        s = cleaned
    return s


def _looks_like_end_matter(text: str) -> bool:
    t = _clean_text(text).lower()
    bad_markers = [
        "references",
        "acknowledg",
        "bibliography",
        "author contributions",
        "competing interests",
        "copyright",
        "licensee american",
    ]
    return any(x in t for x in bad_markers)


def _looks_like_author_line(text: str) -> bool:
    t = _clean_text(text)
    if not t:
        return False
    if "*" in t or "†" in t:
        return True
    tokens = t.split()
    cap_words = sum(1 for w in tokens if w[:1].isupper())
    return len(tokens) >= 4 and cap_words >= max(3, len(tokens) // 2) and "," in t


def _is_section_label(text: str) -> bool:
    t = _clean_text(text)
    if not t:
        return False
    return len(t.split()) <= 5 and t.upper() == t and len(re.sub(r"[^A-Za-z]", "", t)) >= 4


def _is_spaced_banner(text: str) -> bool:
    tokens = _clean_text(text).split()
    if len(tokens) < 4:
        return False
    single_char = sum(1 for tok in tokens if len(tok) == 1 and tok.isalpha() and tok.upper() == tok)
    return single_char >= max(4, int(0.6 * len(tokens)))


def extract_title_candidates(passage_items, max_candidates=8):
    scored = []
    for item in passage_items:
        text = _clean_text(item.get("text_for_embed", ""))
        page = item.get("page")
        chunk_idx = int(item.get("chunk_idx", 10**6))
        block_type = (item.get("block_type") or "").lower()
        if block_type in {"footer", "page_number"}:
            continue
        if not text or _looks_like_end_matter(text) or _looks_like_author_line(text) or _is_section_label(text) or _is_spaced_banner(text):
            continue
        words = text.split()
        score = 0.0
        if page == 1:
            score += 3.0
        elif page == 2:
            score += 1.5
        if chunk_idx <= 2:
            score += 2.0
        elif chunk_idx <= 6:
            score += 1.0
        if block_type == "title":
            score += 4.0
        if 6 <= len(words) <= 22:
            score += 2.5
        elif 4 <= len(words) <= 28:
            score += 1.0
        if re.search(r"[A-Za-z]", text):
            score += 0.5
        if text.endswith("."):
            score -= 0.8
        if re.search(r"\b(?:et al\.|sci\. adv\.|copyright|page \d+ of \d+|no claim to)\b", text, flags=re.IGNORECASE):
            score -= 2.5
        scored.append(
            {
                "candidate": text,
                "page": page,
                "chunk_idx": chunk_idx,
                "block_type": block_type,
                "score": score,
            }
        )
    scored.sort(key=lambda x: (-x["score"], x["page"] if isinstance(x["page"], int) else 10**9, x["chunk_idx"]))
    return scored[:max_candidates]


def extract_title_answer(passage_items):
    candidates = extract_title_candidates(passage_items)
    if not candidates:
        return "", candidates
    return candidates[0]["candidate"], candidates


def normalize_figure_token(token: str) -> str:
    token = re.sub(r"\s+", " ", (token or "")).strip()
    token = token.replace("Figure", "Fig")
    token = token.replace("Fig.", "Fig")
    token = token.replace("Fig ", "Fig ")
    if token.lower().startswith("fig"):
        m = re.search(r"(?:\b| )(S)?\s*(\d+)", token, flags=re.IGNORECASE)
        if not m:
            return token
        prefix = "Fig. "
        if m.group(1):
            return f"{prefix}S{m.group(2)}"
        return f"{prefix}{m.group(2)}"
    if token.lower().startswith("table"):
        m = re.search(r"(?:\b| )(S)?\s*(\d+)", token, flags=re.IGNORECASE)
        if not m:
            return token
        if m.group(1):
            return f"Table S{m.group(2)}"
        return f"Table {m.group(2)}"
    return token


def is_supplementary_token(token: str) -> bool:
    return bool(re.search(r"\b(?:Fig\.?|Table)\s*S\d+\b", normalize_figure_token(token), flags=re.IGNORECASE))


def extract_figure_token_candidates(passage_items, max_candidates=12):
    candidates = []
    ordered = sorted(
        passage_items,
        key=lambda x: (
            x.get("page") if isinstance(x.get("page"), int) else 10**9,
            int(x.get("chunk_idx", 10**6)),
        ),
    )
    for item in ordered:
        text = _clean_text(item.get("text_for_embed", ""))
        if not text or _looks_like_end_matter(text):
            continue
        for match in FIG_TOKEN_RE.finditer(text):
            token = normalize_figure_token(match.group(1))
            candidates.append(
                {
                    "token": token,
                    "page": item.get("page"),
                    "chunk_idx": item.get("chunk_idx"),
                    "anchors": item.get("anchors", []),
                    "supplementary": is_supplementary_token(token),
                    "snippet": text[:180],
                }
            )
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def extract_first_figure_answer(passage_items):
    candidates = extract_figure_token_candidates(passage_items)
    if not candidates:
        return "", candidates
    for cand in candidates:
        if not cand["supplementary"]:
            return cand["token"], candidates
    return candidates[0]["token"], candidates


def _split_sentences(text: str):
    text = _clean_text(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _looks_like_quant_statement(sentence: str) -> bool:
    s = _clean_text(sentence)
    if not s or _looks_like_end_matter(s):
        return False
    if s.lower().startswith(("fig.", "table", "figure")):
        return False
    return bool(re.search(r"\d", s) and QUANT_MARKER_RE.search(s))


def extract_quant_candidates(passage_items, max_candidates=8):
    candidates = []
    ordered = sorted(
        passage_items,
        key=lambda x: (
            x.get("page") if isinstance(x.get("page"), int) else 10**9,
            int(x.get("chunk_idx", 10**6)),
        ),
    )
    for item in ordered:
        text = _clean_text(item.get("text_for_embed", ""))
        page = item.get("page")
        chunk_idx = int(item.get("chunk_idx", 10**6))
        block_type = (item.get("block_type") or "").lower()
        if not text or _looks_like_end_matter(text) or block_type in {"footer", "page_number", "title"}:
            continue
        for sentence in _split_sentences(text):
            if not _looks_like_quant_statement(sentence):
                continue
            score = 0.0
            if isinstance(page, int):
                if page == 1:
                    score += 3.0
                elif page == 2:
                    score += 2.0
                elif page <= 4:
                    score += 1.0
            if chunk_idx <= 8:
                score += 1.5
            if "%" in sentence or "percent" in sentence.lower():
                score += 1.0
            if re.search(r"\b(up to|turnover number|l/b ratio|yield|increased by|decreased by)\b", sentence, flags=re.IGNORECASE):
                score += 1.5
            if len(sentence.split()) >= 8:
                score += 0.5
            candidates.append(
                {
                    "statement": sentence,
                    "page": page,
                    "chunk_idx": chunk_idx,
                    "block_type": block_type,
                    "score": score,
                }
            )
    candidates.sort(key=lambda x: (-x["score"], x["page"] if isinstance(x["page"], int) else 10**9, x["chunk_idx"]))
    return candidates[:max_candidates]


def extract_quant_plus_figure_answer(passage_items):
    figure_token, figure_token_candidates = extract_first_figure_answer(passage_items)
    quant_candidates = extract_quant_candidates(passage_items)
    chosen_quant_statement = quant_candidates[0]["statement"] if quant_candidates else ""
    final_answer = ""
    if chosen_quant_statement and figure_token:
        final_answer = f"Key statement: {chosen_quant_statement} First figure/table: {figure_token}"
    elif figure_token:
        final_answer = f"Key statement: not found. First figure/table: {figure_token}"
    elif chosen_quant_statement:
        final_answer = f"Key statement: {chosen_quant_statement} First figure/table: not found"
    return final_answer, quant_candidates, chosen_quant_statement, figure_token, figure_token_candidates


def _looks_like_summary_lead(text: str) -> bool:
    t = _clean_text(text).lower()
    markers = [
        "here, we",
        "here we",
        "we demonstrate",
        "we present",
        "we report",
        "this work",
        "this paper",
        "this study",
        "we identify",
        "we further show",
        "we herein report",
        "we herein reported",
        "to resolve",
    ]
    return any(m in t for m in markers)


def _truncate_to_one_sentence(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    first = parts[0].strip()
    if first and not first.endswith((".", "!", "?")):
        first = first.rstrip(" ;,:")
        first = f"{first}."
    return first


def _extract_best_summary_sentence(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return ""
    for sentence in sentences:
        if _looks_like_summary_lead(sentence):
            if not sentence.endswith((".", "!", "?")):
                sentence = f"{sentence.rstrip(' ;,:')}."
            return sentence
    return _truncate_to_one_sentence(text)


def extract_summary_candidates(passage_items, max_candidates=6):
    scored = []
    ordered = sorted(
        passage_items,
        key=lambda x: (
            x.get("page") if isinstance(x.get("page"), int) else 10**9,
            int(x.get("chunk_idx", 10**6)),
        ),
    )
    for item in ordered:
        text = _clean_text(item.get("text_for_embed", ""))
        page = item.get("page")
        chunk_idx = int(item.get("chunk_idx", 10**6))
        block_type = (item.get("block_type") or "").lower()
        if block_type in {"footer", "page_number"}:
            continue
        if not text or _looks_like_end_matter(text) or _looks_like_author_line(text) or _is_spaced_banner(text):
            continue
        words = text.split()
        score = 0.0
        if isinstance(page, int):
            if page == 1:
                score += 3.0
            elif page == 2:
                score += 2.0
            elif page == 3:
                score += 0.5
        if chunk_idx <= 6:
            score += 1.5
        if block_type == "paragraph":
            score += 1.0
        if block_type == "title":
            score -= 2.5
        if _looks_like_summary_lead(text):
            score += 3.0
        if 15 <= len(words) <= 120:
            score += 1.0
        elif len(words) >= 12:
            score += 0.5
        elif len(words) <= 8:
            score -= 1.5
        if text.endswith("."):
            score += 0.3
        scored.append(
            {
                "candidate": text,
                "page": page,
                "chunk_idx": chunk_idx,
                "block_type": block_type,
                "score": score,
            }
        )
    scored.sort(key=lambda x: (-x["score"], x["page"] if isinstance(x["page"], int) else 10**9, x["chunk_idx"]))
    return scored[:max_candidates]


def extract_summary_answer(passage_items):
    candidates = extract_summary_candidates(passage_items)
    if not candidates:
        return "", candidates
    for cand in candidates:
        sentence = _extract_best_summary_sentence(cand["candidate"])
        if sentence:
            return sentence, candidates
    return "", candidates


def select_doc_passage_items(passage_metadata_by_hash, doc_id):
    doc_id = normalize_doc_id(doc_id)
    out = []
    for hid, meta in passage_metadata_by_hash.items():
        if normalize_doc_id(meta.get("doc_id", "")) != doc_id:
            continue
        row = dict(meta)
        row["hash_id"] = hid
        out.append(row)
    return out
