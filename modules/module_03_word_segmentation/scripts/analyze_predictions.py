"""Create a dev-first error analysis from model predictions and gold rows."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "external" / "Hackathon_dcs.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "experiments" / "module_03_segmentation" / "character_transformer" / "dev_predictions.csv"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "module_03_segmentation" / "error_analysis"


def word_scores(prediction, reference):
    predicted = prediction.split()
    gold = reference.split()
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    precision = overlap / len(predicted) if predicted else float(not gold)
    recall = overlap / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--review-limit", type=int, default=200)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype="string")
    source_rows = frame[frame["split"].eq(args.split)].drop_duplicates(["input", "output"])
    references = defaultdict(list)
    for row in source_rows[["input", "output"]].itertuples(index=False):
        references[row.input].append(row.output)
    predictions = pd.read_csv(args.predictions, encoding="utf-8-sig", dtype="string").fillna("")
    if "input" not in predictions or "prediction" not in predictions:
        raise ValueError("Prediction file must contain input and prediction columns.")
    if predictions["input"].duplicated().any():
        raise ValueError("Expected one prediction per unique input.")

    records = []
    totals = Counter()
    macro = {key: 0.0 for key in ("precision", "recall", "f1")}
    for row in predictions.to_dict("records"):
        source = row["input"]
        if source not in references:
            raise ValueError(f"Prediction input is absent from the requested {args.split} split.")
        prediction = row["prediction"]
        scored = [(word_scores(prediction, reference), reference) for reference in references[source]]
        (precision, recall, f1), gold = max(scored, key=lambda item: (item[0][2], prediction == item[1]))
        exact = prediction == gold
        prediction_chars = "".join(prediction.split())
        gold_chars = "".join(gold.split())
        source_chars = "".join(source.split())
        char_error = Levenshtein.distance(prediction_chars, gold_chars)
        boundary_only = not exact and prediction_chars == gold_chars
        bucket = (
            "0-30" if len(source) <= 30 else
            "31-45" if len(source) <= 45 else
            "46-60" if len(source) <= 60 else
            "61-90" if len(source) <= 90 else "91+"
        )
        totals["exact"] += int(exact)
        totals["boundary_only_mismatch"] += int(boundary_only)
        totals["character_sequence_mismatch"] += int(not exact and not boundary_only)
        totals["fallback"] += int(str(row.get("fallback_applied", "false")).lower() == "true")
        macro["precision"] += precision
        macro["recall"] += recall
        macro["f1"] += f1
        records.append({
            "input": source, "prediction": prediction, "reference": gold,
            "exact": exact, "word_precision": precision, "word_recall": recall,
            "word_f1": f1, "input_characters": len(source),
            "prediction_words": len(prediction.split()), "reference_words": len(gold.split()),
            "character_edit_distance_without_whitespace": char_error,
            "input_prediction_character_edit_ratio": Levenshtein.distance(source_chars, prediction_chars) / max(len(source_chars), 1),
            "length_bucket": bucket,
            "fallback_applied": str(row.get("fallback_applied", "false")).lower() == "true",
        })

    if len(records) != len(references):
        raise ValueError(f"Expected one prediction for each unique split input: {len(records)} != {len(references)}")
    count = max(len(records), 1)
    by_bucket = {}
    for bucket in ("0-30", "31-45", "46-60", "61-90", "91+"):
        selected = [row for row in records if row["length_bucket"] == bucket]
        if selected:
            by_bucket[bucket] = {
                "inputs": len(selected),
                "exact_match_rate": sum(row["exact"] for row in selected) / len(selected),
                "macro_word_f1": sum(row["word_f1"] for row in selected) / len(selected),
            }

    report = {
        "split": args.split,
        "unique_inputs": len(records),
        "metrics": {
            "macro_word_precision": macro["precision"] / count,
            "macro_word_recall": macro["recall"] / count,
            "macro_word_f1": macro["f1"] / count,
            "perfect_sentence_match": totals["exact"] / count,
        },
        "error_types": {
            "exact": totals["exact"],
            "boundary_only_mismatch": totals["boundary_only_mismatch"],
            "character_sequence_mismatch": totals["character_sequence_mismatch"],
            "identity_fallbacks": totals["fallback"],
        },
        "performance_by_input_length": by_bucket,
        "review_sample_policy": f"Up to {args.review_limit} non-exact rows ranked by character edit distance; local use only.",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / f"{args.split}_summary.json"
    review_path = OUTPUT_ROOT / f"{args.split}_review_examples.csv"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    errors = sorted((row for row in records if not row["exact"]),
                    key=lambda row: (row["character_edit_distance_without_whitespace"], row["word_f1"]),
                    reverse=True)[:args.review_limit]
    if errors:
        with review_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(errors[0]))
            writer.writeheader()
            writer.writerows(errors)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Summary: {summary_path}")
    if errors:
        print(f"Review examples: {review_path}")


if __name__ == "__main__":
    main()
