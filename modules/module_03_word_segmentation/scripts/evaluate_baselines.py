"""Evaluate whitespace and train-memorization baselines on Hackathon DCS."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "external" / "Hackathon_dcs.csv"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "module_03_segmentation" / "baseline_metrics.json"


def score(prediction, references):
    predicted = prediction.split()
    best = (0.0, 0.0, 0.0, False, references[0])
    for reference in references:
        gold = reference.split()
        overlap = sum((Counter(predicted) & Counter(gold)).values())
        precision = overlap / len(predicted) if predicted else float(not gold)
        recall = overlap / len(gold) if gold else float(not predicted)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, (precision, recall, f1, prediction == reference, reference), key=lambda x: (x[2], x[3]))
    return best


def evaluate(rows, predictor):
    references = defaultdict(list)
    for row in rows:
        if row["output"] not in references[row["input"]]:
            references[row["input"]].append(row["output"])
    macro_p = macro_r = macro_f1 = 0.0
    exact = 0
    micro_overlap = micro_pred = micro_gold = 0
    for source, targets in references.items():
        prediction = predictor(source)
        p, r, f1, exact_any, selected_reference = score(prediction, targets)
        macro_p += p
        macro_r += r
        macro_f1 += f1
        exact += int(exact_any)
        pred = Counter(prediction.split())
        gold = Counter(selected_reference.split())
        micro_overlap += sum((pred & gold).values())
        micro_pred += sum(pred.values())
        micro_gold += sum(gold.values())
    count = max(1, len(references))
    micro_p = micro_overlap / micro_pred if micro_pred else 0.0
    micro_r = micro_overlap / micro_gold if micro_gold else 0.0
    return {
        "rows": len(rows), "unique_inputs": len(references),
        "macro_word_precision": macro_p / count,
        "macro_word_recall": macro_r / count,
        "macro_word_f1": macro_f1 / count,
        "micro_word_precision": micro_p, "micro_word_recall": micro_r,
        "micro_word_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0,
        "perfect_sentence_match_any_reference": exact / count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype="string")
    if set(frame.columns) != {"input", "output", "split", "DCS-ID"}:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    rows = frame.to_dict("records")
    train = [r for r in rows if r["split"] == "train"]
    most_common = defaultdict(Counter)
    for row in train:
        most_common[row["input"]][row["output"]] += 1
    memorized = {source: counts.most_common(1)[0][0] for source, counts in most_common.items()}
    result = {"dataset": str(args.input), "baselines": {}}
    for split in ("dev", "test"):
        current = [r for r in rows if r["split"] == split]
        result["baselines"][split] = {
            "identity_input": evaluate(current, lambda source: source),
            "train_most_frequent_or_identity": evaluate(current, lambda source: memorized.get(source, source)),
        }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["baselines"], indent=2, ensure_ascii=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
