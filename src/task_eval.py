from src.text_metrics import exact_match_score, token_f1_score


def extract_gold_payload(question_info):
    if not isinstance(question_info, dict):
        return ""
    if "answer" in question_info:
        return question_info["answer"]
    gold = question_info.get("gold")
    if isinstance(gold, dict):
        return gold
    return ""


def extract_gold_answer_text(gold_answer, question_type=""):
    if isinstance(gold_answer, dict):
        if question_type == "summary":
            return gold_answer.get("summary_one_sentence") or gold_answer.get("canonical_answer", "")
        if question_type == "quant_plus_first_figure":
            return gold_answer.get("canonical_answer", "")
        return gold_answer.get("canonical_answer") or gold_answer.get("title") or gold_answer.get("token") or ""
    return gold_answer or ""


def evaluate_quant_prediction(pred_answer, gold_answer):
    if not isinstance(gold_answer, dict):
        return {
            "token_match": 0.0,
            "quant_token_f1": 0.0,
            "canonical_answer_em": 0.0,
        }
    pred_answer = pred_answer or ""
    token = gold_answer.get("token", "")
    quant_statement = gold_answer.get("quant_statement", "")
    canonical_answer = gold_answer.get("canonical_answer", "")
    token_match = 1.0 if token and f"first figure/table: {token}".lower() in pred_answer.lower() else 0.0
    quant_part = pred_answer
    lower_pred = pred_answer.lower()
    if "first figure/table:" in lower_pred:
        quant_part = pred_answer[: lower_pred.index("first figure/table:")].strip()
    quant_part = quant_part.replace("Key statement:", "").strip()
    return {
        "token_match": token_match,
        "quant_token_f1": token_f1_score(quant_part, quant_statement),
        "canonical_answer_em": exact_match_score(pred_answer, canonical_answer),
    }


def compute_task_metrics(predictions):
    title_scores = []
    first_figure_scores = []
    summary_scores = []
    quant_token_match_scores = []
    quant_token_f1_scores = []
    quant_canonical_em_scores = []

    for pred in predictions:
        qtype = pred.get("question_type", "")
        pred_answer = pred.get("pred_answer", "")
        gold_answer = pred.get("gold_answer")
        gold_text = extract_gold_answer_text(gold_answer, qtype)
        pred["gold_answer_text"] = gold_text

        if qtype == "title":
            score = exact_match_score(pred_answer, gold_text)
            pred["title_em"] = score
            title_scores.append(score)
        elif qtype == "first_figure_token":
            score = exact_match_score(pred_answer, gold_text)
            pred["first_figure_token_em"] = score
            first_figure_scores.append(score)
        elif qtype == "summary":
            score = token_f1_score(pred_answer, gold_text)
            pred["summary_token_f1"] = score
            summary_scores.append(score)
        elif qtype == "quant_plus_first_figure":
            metrics = evaluate_quant_prediction(pred_answer, gold_answer)
            pred.update(metrics)
            quant_token_match_scores.append(metrics["token_match"])
            quant_token_f1_scores.append(metrics["quant_token_f1"])
            quant_canonical_em_scores.append(metrics["canonical_answer_em"])

    return {
        "title_em": (sum(title_scores) / len(title_scores)) if title_scores else 0.0,
        "first_figure_token_em": (sum(first_figure_scores) / len(first_figure_scores)) if first_figure_scores else 0.0,
        "summary_token_f1": (sum(summary_scores) / len(summary_scores)) if summary_scores else 0.0,
        "quant_token_match": (sum(quant_token_match_scores) / len(quant_token_match_scores)) if quant_token_match_scores else 0.0,
        "quant_token_f1": (sum(quant_token_f1_scores) / len(quant_token_f1_scores)) if quant_token_f1_scores else 0.0,
        "quant_canonical_answer_em": (sum(quant_canonical_em_scores) / len(quant_canonical_em_scores)) if quant_canonical_em_scores else 0.0,
    }
