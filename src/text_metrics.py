import re
import string
from collections import Counter


def normalize_text(text):
    text = "" if text is None else str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def exact_match_score(pred, gold):
    p = normalize_text(pred)
    g = normalize_text(gold)
    return 1.0 if p and p == g else 0.0


def contain_score(pred, gold):
    p = normalize_text(pred)
    g = normalize_text(gold)
    return 1.0 if p and g and g in p else 0.0


def token_f1_score(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
