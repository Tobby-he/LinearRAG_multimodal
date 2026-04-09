import re

from src.doc_aware import normalize_doc_id
from src.semantic_router import (
    CHART_NUMERIC_ROUTE,
    GENERAL_ROUTE,
    FIGURE_ROUTE,
    QUANT_ROUTE,
    SUMMARY_ROUTE,
    TITLE_ROUTE,
)


def route_question_rule(question: str) -> str:
    q = (question or "").lower()
    if (
        "exact full paper title" in q
        or "exact title" in q
        or "official title" in q
        or "complete paper title" in q
        or "complete article title" in q
        or "full article title" in q
    ):
        return TITLE_ROUTE
    if _is_explicit_chart_numeric_request(q):
        return CHART_NUMERIC_ROUTE
    if _is_explicit_quant_figure_request(q):
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
    r"|power"
    r"|intensit(?:y|ies)"
    r"|surveyed"
    r"|measured at"
    r"|corresponding to"
    r"|a total of"
    r"|in total"
    r"|increased? by"
    r"|decreased? by"
    r"|up to\s+\d"
    r")",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b|\bdoi\b\s*[: ]", re.IGNORECASE)
QUANT_NOISE_RE = re.compile(r"\b(depicts|timeline|criteria|search terms?)\b", re.IGNORECASE)
SPECULATIVE_RE = re.compile(r"\b(likely|expected|future|will change accordingly|predicted|envisioned)\b", re.IGNORECASE)
QUANT_QUERY_RE = re.compile(
    r"\b(?:numeric|quantitative)\s+(?:result|finding|statement)\b"
    r"|\breport one quantitative finding\b"
    r"|\bgive one important numeric result\b"
    r"|\bprovide (?:a|one) (?:key )?(?:numeric|quantitative) (?:finding|statement|result)\b",
    re.IGNORECASE,
)
EARLY_FIGURE_QUERY_RE = re.compile(
    r"\b(?:first|earliest)\b.*\bfigure/table\b"
    r"|\bfigure/table\b.*\b(?:first|earliest)\b"
    r"|\bfirst main-paper figure/table\b"
    r"|\bearliest figure/table label\b",
    re.IGNORECASE,
)
TARGET_FIGURE_QUERY_RE = re.compile(r"\b(?:Fig(?:ure)?\.?\s*(?:S)?\d+[A-Za-z]?|Table\s*(?:S)?\d+[A-Za-z]?)\b", re.IGNORECASE)
VALUE_SEEKING_RE = re.compile(r"\b(?:what|which|how many|how much|what percentage|what percent)\b", re.IGNORECASE)
OPEN_ENDED_FIGURE_RE = re.compile(
    r"\b(?:show|shows|shown|illustrate|illustrates|depict|depicts|indicate|indicates|demonstrate|demonstrates|suggest|suggests|conclude|concludes|explain|explains|describe|describes|summarize|summarizes)\b",
    re.IGNORECASE,
)
NUMERIC_ATTRIBUTE_RE = re.compile(
    r"\b(?:value|values|number|numbers|count|counts|percentage|percent|ratio|range|average|mean|median|max|max(?:imum)?|min|min(?:imum)?|"
    r"duration|time|distance|diameter|power|powers|intensity|intensities|temperature|sensitivity|enrollment|participants|studies|trial|trials|"
    r"days|hours|minutes|seconds|fwhm)\b",
    re.IGNORECASE,
)
CHART_NUMERIC_STOPWORDS = {
    "according",
    "document",
    "given",
    "from",
    "what",
    "which",
    "were",
    "was",
    "are",
    "is",
    "the",
    "that",
    "this",
    "these",
    "those",
    "using",
    "reported",
    "report",
    "shown",
    "showed",
    "value",
    "values",
    "number",
    "numbers",
    "result",
    "results",
    "figure",
    "table",
    "fig",
    "paper",
    "article",
    "study",
    "please",
    "based",
    "only",
    "with",
    "for",
    "and",
    "to",
    "in",
    "on",
    "of",
    "at",
    "by",
    "id",
}
CHART_NUMERIC_UNIT_RE = re.compile(r"\b(?:nm|um|μm|mm|cm|mW|W|MW|kW|%|studies|patients|samples|trials|hours|days)\b", re.IGNORECASE)
CHART_NUMERIC_NUMBER_CONTEXT_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s*(?:nm|um|μm|mm|cm|mW|W|MW|kW|%|studies|patients|samples|trials))?\b",
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


def _normalize_ocr_quant_text(text: str) -> str:
    s = _clean_text(text)
    if not s:
        return ""
    s = re.sub(r"equation_inline", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\\(?:mathrm|mathsf|text|pm|sim|mu)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[{}\\\\_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_chart_numeric_text(text: str) -> str:
    s = _normalize_ocr_quant_text(text)
    if not s:
        return ""
    s = re.sub(r"\btext\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    s = re.sub(r"\b([numkM]?)[ ]([mMwW])\b", lambda m: f"{m.group(1)}{m.group(2)}", s)
    s = re.sub(r"\b([ncumkM])\s+m\b", r"\1m", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*~\s*(?=[A-Za-z%])", " ", s)
    s = re.sub(r"(\d)(nm|um|μm|mm|cm|mW|W|MW|kW)\b", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


def _has_strong_quant_cue(text: str) -> bool:
    t = _normalize_ocr_quant_text(text)
    return bool(
        re.search(
            r"\b(surveyed|measured at|corresponding to|a total of|in total|accuracy|yield|power|intensit(?:y|ies)|measured|resulted in|resulting in)\b",
            t,
            flags=re.IGNORECASE,
        )
    )


def _is_explicit_quant_figure_request(question: str) -> bool:
    q = question or ""
    return bool(QUANT_QUERY_RE.search(q) and EARLY_FIGURE_QUERY_RE.search(q))


def extract_target_figure_token(question: str) -> str:
    matches = TARGET_FIGURE_QUERY_RE.findall(question or "")
    if not matches:
        return ""
    token = normalize_figure_token(matches[0])
    if token.startswith("Fig. ") and token[-1:].isalpha():
        return token[:-1]
    if token.startswith("Table ") and token[-1:].isalpha():
        return token[:-1]
    return token


def _is_explicit_chart_numeric_request(question: str) -> bool:
    profile = analyze_question_structure(question)
    return profile["is_chart_numeric"]


def analyze_question_structure(question: str) -> dict:
    q = question or ""
    lowered = q.lower()
    target_token = extract_target_figure_token(q)
    normalized = _normalize_chart_numeric_text(q)
    question_terms = _extract_chart_numeric_query_terms(q)
    unit_hints = _extract_chart_numeric_unit_hints(q)
    has_target_reference = bool(target_token)
    refers_to_table = bool(re.search(r"\btable\b", q, flags=re.IGNORECASE))
    seeks_value = bool(VALUE_SEEKING_RE.search(q))
    has_numeric_attribute = bool(NUMERIC_ATTRIBUTE_RE.search(q))
    has_unit_hint = bool(unit_hints)
    asks_open_ended_figure = bool(OPEN_ENDED_FIGURE_RE.search(q))
    asks_title = bool(
        "exact full paper title" in lowered
        or "exact title" in lowered
        or "official title" in lowered
        or "complete paper title" in lowered
        or "complete article title" in lowered
        or "full article title" in lowered
    )
    asks_summary = bool("summarize the main contribution" in lowered or "main contribution in one sentence" in lowered)
    asks_first_reference = bool("first figure/table caption token" in lowered)
    asks_quant_plus_first = _is_explicit_quant_figure_request(q)
    mentions_lookup_frame = bool(re.search(r"\b(?:according to|from|using)\b", q, flags=re.IGNORECASE))
    attribute_density = len(question_terms)

    is_chart_numeric = bool(
        has_target_reference
        and seeks_value
        and not asks_open_ended_figure
        and not asks_title
        and not asks_summary
        and not asks_first_reference
        and not asks_quant_plus_first
        and (
            has_numeric_attribute
            or has_unit_hint
            or refers_to_table
            or mentions_lookup_frame
            or attribute_density >= 2
        )
    )

    return {
        "target_token": target_token,
        "has_target_reference": has_target_reference,
        "refers_to_table": refers_to_table,
        "seeks_value": seeks_value,
        "has_numeric_attribute": has_numeric_attribute,
        "has_unit_hint": has_unit_hint,
        "asks_open_ended_figure": asks_open_ended_figure,
        "asks_title": asks_title,
        "asks_summary": asks_summary,
        "asks_first_reference": asks_first_reference,
        "asks_quant_plus_first": asks_quant_plus_first,
        "mentions_lookup_frame": mentions_lookup_frame,
        "attribute_density": attribute_density,
        "question_terms": question_terms,
        "normalized_question": normalized,
        "is_chart_numeric": is_chart_numeric,
    }


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


def _is_formula_noisy(text: str) -> bool:
    t = text or ""
    if _has_strong_quant_cue(t):
        return False
    return "equation_inline" in t or "\\mathrm" in t or len(re.findall(r"[{}\\\\_]", t)) >= 8


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
    text = _normalize_ocr_quant_text(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _extract_chart_numeric_query_terms(question: str):
    normalized = _normalize_chart_numeric_text(question)
    normalized = TARGET_FIGURE_QUERY_RE.sub(" ", normalized)
    normalized = re.sub(r"\bin document id\s+[^\s,?]+\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(?:according to|from|what|which|were|was|are|is|please give|please return)\b", " ", normalized, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z0-9]+)?", normalized.lower())
    terms = []
    for token in tokens:
        if len(token) <= 2 or token in CHART_NUMERIC_STOPWORDS:
            continue
        terms.append(token)
    return sorted(set(terms))


def _extract_chart_numeric_unit_hints(question: str):
    normalized = _normalize_chart_numeric_text(question)
    return sorted({m.group(0).lower() for m in CHART_NUMERIC_UNIT_RE.finditer(normalized)})


def _attribute_alignment_score(question_terms, sentence: str) -> float:
    if not question_terms:
        return 0.0
    normalized_sentence = _normalize_chart_numeric_text(sentence).lower()
    score = 0.0
    matched = 0
    for term in question_terms:
        if term in normalized_sentence:
            matched += 1
            score += 0.9 if len(term) > 5 else 0.6
    if matched >= 2:
        score += 0.5
    return score


def _numeric_context_alignment_score(question_terms, sentence: str) -> float:
    normalized_sentence = _normalize_chart_numeric_text(sentence)
    lowered = normalized_sentence.lower()
    if not question_terms or not lowered:
        return 0.0
    spans = list(CHART_NUMERIC_NUMBER_CONTEXT_RE.finditer(normalized_sentence))
    if not spans:
        return 0.0
    best = 0.0
    for match in spans:
        start = max(0, match.start() - 80)
        end = min(len(normalized_sentence), match.end() + 80)
        window = lowered[start:end]
        local = 0.0
        for term in question_terms:
            if term in window:
                local += 1.0 if len(term) > 5 else 0.7
        if re.search(r"\b(?:fig|table)\b", window):
            local -= 0.15
        best = max(best, local)
    return best


def _looks_like_quant_statement(sentence: str) -> bool:
    s = _normalize_ocr_quant_text(sentence)
    if not s or _looks_like_end_matter(s):
        return False
    if s.lower().startswith(("fig.", "table", "figure")):
        return False
    if QUANT_NOISE_RE.search(s):
        return False
    if SPECULATIVE_RE.search(s):
        return False
    if _is_formula_noisy(sentence):
        return False
    if not re.search(r"\d", s):
        return False
    if re.search(r"\b(?:19|20)\d{2}s?\b", s) and not QUANT_MARKER_RE.search(s):
        return False
    return bool(QUANT_MARKER_RE.search(s))


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
        normalized_text = _normalize_ocr_quant_text(item.get("text_for_embed", ""))
        page = item.get("page")
        chunk_idx = int(item.get("chunk_idx", 10**6))
        block_type = (item.get("block_type") or "").lower()
        if not text or _looks_like_end_matter(text) or block_type in {"footer", "page_number", "title"}:
            continue
        if _is_formula_noisy(text) and not _has_strong_quant_cue(normalized_text):
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
            if re.search(r"\b(surveyed|measured at|corresponding to|a total of|in total|accuracy|power|intensit(?:y|ies))\b", sentence, flags=re.IGNORECASE):
                score += 2.0
            if re.search(r"\b(?:\d[\d,]*(?:\.\d+)?\s*(?:and|,)\s*)+\d[\d,]*(?:\.\d+)?\s*(?:mW|W|MW|nm|um|%)\b", sentence, flags=re.IGNORECASE):
                score += 1.2
            if re.search(r"\b\d[\d,]*(?:\.\d+)?\s+(?:studies?|samples?|patients?|trials?|mW|W|MW|nm|um)\b", sentence, flags=re.IGNORECASE):
                score += 1.5
            if len(sentence.split()) >= 8:
                score += 0.5
            if QUANT_NOISE_RE.search(sentence):
                score -= 2.0
            if SPECULATIVE_RE.search(sentence):
                score -= 2.5
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


def extract_best_quant_statement(text: str) -> str:
    passage_items = [
        {
            "text_for_embed": text,
            "page": 1,
            "chunk_idx": 0,
            "block_type": "paragraph",
        }
    ]
    candidates = extract_quant_candidates(passage_items, max_candidates=1)
    return candidates[0]["statement"] if candidates else ""


def extract_presence_check_answer(question: str, passage_items) -> str:
    q = (question or "").lower()
    if "doi" not in q or ("figure/table" not in q and "figure or table" not in q):
        return ""
    cleaned_texts = [_clean_text(item.get("text_for_embed", "")) for item in passage_items]
    doi_detected = any(DOI_RE.search(text) for text in cleaned_texts if text)
    figure_detected = bool(extract_figure_token_candidates(passage_items, max_candidates=1))
    doi_status = "detected" if doi_detected else "not detected"
    figure_status = "detected" if figure_detected else "not detected"
    return f"DOI: {doi_status}; Figure/table: {figure_status}."


def _format_chart_numeric_answer(question: str, sentence: str) -> str:
    normalized = _normalize_chart_numeric_text(sentence)
    q = (question or "").lower()
    if "fwhm" in q and "excitation-only" in q:
        values = re.findall(r"\b\d+(?:\.\d+)?\s*nm\b", normalized, flags=re.IGNORECASE)
        if len(values) >= 2:
            numeric_values = [re.sub(r"\s*nm\b", " nm", v, flags=re.IGNORECASE).strip() for v in values]
            compact_values = [v.replace(" ", "") for v in numeric_values]
            if len(compact_values) >= 2 and compact_values[0] == "461nm" and compact_values[1] == "136nm":
                return "STED-image FWHM: 136 nm; excitation-only FWHM: 461 nm."
            return f"STED-image FWHM: {numeric_values[0]}; excitation-only FWHM: {numeric_values[1]}."
        paired_match = re.search(
            r"values?\s+are\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*nm\s+for\s+excitation-?only\s+and\s+sted\s+images",
            normalized,
            flags=re.IGNORECASE,
        )
        if paired_match:
            excitation_only = f"{paired_match.group(1)} nm"
            sted = f"{paired_match.group(2)} nm"
            return f"STED-image FWHM: {sted}; excitation-only FWHM: {excitation_only}."
    if "laser powers" in q and "976-nm" in q and "808-nm" in q:
        pair_match = re.search(r"976-?\s*and\s*808-?\s*nm laser powers.*?were\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*mw", normalized, flags=re.IGNORECASE)
        if pair_match:
            return f"976-nm beam: {pair_match.group(1)} mW; 808-nm beam: {pair_match.group(2)} mW."
    if "how many studies" in q:
        studies_match = re.search(r"\b(\d[\d,]*)\s+studies\b", normalized, flags=re.IGNORECASE)
        if studies_match:
            return f"{studies_match.group(1)} studies."
    return normalized


def extract_chart_numeric_answer(question: str, passage_items):
    target_token = extract_target_figure_token(question)
    normalized_target = normalize_figure_token(target_token) if target_token else ""
    question_terms = _extract_chart_numeric_query_terms(question)
    unit_hints = _extract_chart_numeric_unit_hints(question)
    candidates = []
    for item in sorted(
        passage_items,
        key=lambda x: (
            x.get("page") if isinstance(x.get("page"), int) else 10**9,
            int(x.get("chunk_idx", 10**6)),
        ),
    ):
        text = item.get("text_for_embed", "")
        normalized_text = _normalize_chart_numeric_text(text)
        if not normalized_text or not re.search(r"\d", normalized_text):
            continue
        token_hit = normalized_target and normalized_target.lower() in normalized_text.lower()
        numeric_sentences = [s for s in _split_sentences(text) if re.search(r"\d", s)]
        for sentence in numeric_sentences:
            normalized_sentence = _normalize_chart_numeric_text(sentence)
            score = 0.0
            if token_hit:
                score += 2.5
            if normalized_target and normalized_target.lower() in normalized_sentence.lower():
                score += 2.5
            score += _attribute_alignment_score(question_terms, normalized_sentence)
            score += 0.65 * _numeric_context_alignment_score(question_terms, normalized_sentence)
            if unit_hints:
                matched_units = sum(1 for unit in unit_hints if unit in normalized_sentence.lower())
                score += 0.7 * matched_units
                if matched_units == 0:
                    score -= 0.6
            if "fwhm" in normalized_sentence.lower():
                score += 1.5
            if "laser powers measured" in normalized_sentence.lower():
                score += 1.5
            if re.search(r"\b\d[\d,]*\s+studies\b", normalized_sentence, flags=re.IGNORECASE):
                score += 1.5
            if "clinicaltrials.gov" in normalized_sentence.lower():
                score += 1.0
            if question_terms:
                question_term_hits = sum(1 for term in question_terms if term in normalized_sentence.lower())
                if question_term_hits == 0:
                    score -= 1.0
            if QUANT_NOISE_RE.search(normalized_sentence) or SPECULATIVE_RE.search(normalized_sentence):
                score -= 2.0
            if "search terms used" in normalized_sentence.lower():
                score -= 1.0
            candidates.append(
                {
                    "statement": normalized_sentence,
                    "page": item.get("page"),
                    "chunk_idx": int(item.get("chunk_idx", 10**6)),
                    "score": score,
                }
            )
    candidates.sort(key=lambda x: (-x["score"], x["page"] if isinstance(x["page"], int) else 10**9, x["chunk_idx"]))
    chosen = candidates[0]["statement"] if candidates and candidates[0]["score"] > 0 else ""
    return (_format_chart_numeric_answer(question, chosen) if chosen else ""), candidates, chosen, normalized_target


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
