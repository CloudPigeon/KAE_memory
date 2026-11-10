"""Offline lexical scoring with explicit failed and missing rows."""
import re
from collections import Counter

from .hotpot_official import normalize_answer, qa_f1_score


def locomo_normalize(text):
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def locomo_f1(prediction, reference):
    predicted = locomo_normalize(prediction).split()
    expected = locomo_normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * overlap / (len(predicted) + len(expected))


def bleu1(prediction, reference):
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    predicted = nltk.word_tokenize(str(prediction).lower())
    expected = nltk.word_tokenize(str(reference).lower())
    return sentence_bleu([expected], predicted, weights=(1, 0, 0, 0),
                         smoothing_function=SmoothingFunction().method1)


def summarize(rows, labels, dataset="locomo", include_bleu=False):
    if dataset not in {"locomo", "hotpotqa"} or not labels:
        raise ValueError("choose a supported dataset and a nonempty label mapping")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) - set(labels):
        raise ValueError("duplicate or unexpected prediction IDs")
    per_category = {}
    values = []
    failed = 0
    missing = 0
    for question_id, label in labels.items():
        row = by_id.get(question_id)
        is_failed = row is None or bool(row.get("failed") or row.get("error"))
        failed += int(is_failed)
        missing += int(row is None)
        prediction = "" if is_failed else str((row.get("result") or {}).get("answer") or "")
        references = label["answer"]
        references = references if isinstance(references, list) else [references]
        if not references:
            raise ValueError("each label needs at least one reference answer")
        references = [str(reference) for reference in references]
        f1_function = locomo_f1 if dataset == "locomo" else qa_f1_score
        normalize = locomo_normalize if dataset == "locomo" else normalize_answer
        scores = {"f1": 0.0, "em": 0.0}
        if include_bleu:
            scores["bleu1"] = 0.0
        if not is_failed:
            scores["f1"] = max(f1_function(prediction, reference) for reference in references)
            scores["em"] = max(float(normalize(prediction) == normalize(reference)) for reference in references)
            if include_bleu:
                scores["bleu1"] = max(bleu1(prediction, reference) for reference in references)
        values.append(scores)
        category = str(label.get("category", "all"))
        per_category.setdefault(category, []).append(scores)

    def aggregate(items):
        return {"n": len(items), **{
            metric: round(100 * sum(item[metric] for item in items) / len(items), 6)
            for metric in items[0]
        }}

    return {**aggregate(values), "failed": failed, "missing": missing,
            "per_category": {category: aggregate(items) for category, items in per_category.items()}}
