"""Summarize the existing Module 1 split for Module 2 tokenizer planning."""

import json
import math
import statistics
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
INPUT_PATH = PROJECT_ROOT / "datasets" / "splits" / "sansar_source_aware_group_splits.parquet"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "tokenization" / "module2_sample_audit.json"
SPLIT_NAMES = ("train", "validation", "test")


def percentile(sorted_values, fraction):
    """Return the nearest-rank percentile from a sorted list."""
    if not sorted_values:
        return 0
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def make_split_summary(rows):
    """Calculate aggregate text and whitespace-token statistics."""
    text_values = [row["normalized_text"] for row in rows]
    valid_texts = [text for text in text_values if isinstance(text, str)]
    character_lengths = sorted(len(text) for text in valid_texts)
    token_lengths = sorted(len(text.split()) for text in valid_texts)

    word_count_rows = [
        row for row in rows
        if isinstance(row.get("normalized_text"), str)
        and row.get("num_words") is not None
    ]
    word_count_matches = sum(
        int(row["num_words"]) == len(row["normalized_text"].split())
        for row in word_count_rows
    )

    return {
        "rows": len(rows),
        "groups": len({row["sampling_group"] for row in rows}),
        "missing_text": sum(not isinstance(text, str) for text in text_values),
        "empty_text": sum(not text.strip() for text in valid_texts),
        "non_nfc_text": sum(
            unicodedata.normalize("NFC", text) != text for text in valid_texts
        ),
        "character_length": {
            "mean": round(statistics.mean(character_lengths), 2) if character_lengths else 0,
            "maximum": max(character_lengths, default=0),
        },
        "whitespace_token_length": {
            "mean": round(statistics.mean(token_lengths), 2) if token_lengths else 0,
            "median": statistics.median(token_lengths) if token_lengths else 0,
            "p90": percentile(token_lengths, 0.90),
            "p95": percentile(token_lengths, 0.95),
            "maximum": max(token_lengths, default=0),
        },
        "source_num_words_matches_whitespace_split": {
            "matches": word_count_matches,
            "comparable_rows": len(word_count_rows),
        },
    }


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing input split: {INPUT_PATH}. Run Module 1's sample split "
            "preparation first; this script does not download or modify data."
        )

    parquet_file = pq.ParquetFile(INPUT_PATH)
    required_columns = {"split", "normalized_text", "sampling_group", "num_words"}
    actual_columns = set(parquet_file.schema_arrow.names)
    missing_columns = required_columns - actual_columns
    if missing_columns:
        raise ValueError(f"Input split is missing required columns: {sorted(missing_columns)}")

    rows_by_split = {split_name: [] for split_name in SPLIT_NAMES}
    for batch in parquet_file.iter_batches(
        columns=sorted(required_columns),
        batch_size=512,
    ):
        for row in batch.to_pylist():
            split_name = row["split"]
            if split_name not in rows_by_split:
                raise ValueError(f"Unexpected split label: {split_name!r}")
            rows_by_split[split_name].append(row)

    target_splits = {}
    group_splits = {}
    for split_name, rows in rows_by_split.items():
        for row in rows:
            text = row["normalized_text"]
            if isinstance(text, str):
                target_splits.setdefault(text, set()).add(split_name)
            group_splits.setdefault(row["sampling_group"], set()).add(split_name)

    report = {
        "input_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "rows_in_file": parquet_file.metadata.num_rows,
        "tokenization_baseline": "Python whitespace split: str.split()",
        "splits": {
            split_name: make_split_summary(rows)
            for split_name, rows in rows_by_split.items()
        },
        "cross_split_exact_text_overlap_count": sum(
            len(split_names) > 1 for split_names in target_splits.values()
        ),
        "cross_split_group_overlap_count": sum(
            len(split_names) > 1 for split_names in group_splits.values()
        ),
        "interpretation": [
            "Whitespace counts describe existing text boundaries; they are not gold Sanskrit word boundaries.",
            "This report contains aggregate statistics only and does not contain Sanskrit text examples.",
            "Tokenizer vocabularies must be fitted on train only; keep validation and test for comparison.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Rows inspected: {report['rows_in_file']:,}")
    print(f"Aggregate audit saved to: {OUTPUT_PATH}")
    for split_name, summary in report["splits"].items():
        lengths = summary["whitespace_token_length"]
        print(
            f"{split_name}: rows={summary['rows']:,}, "
            f"groups={summary['groups']}, median_tokens={lengths['median']}, "
            f"p95_tokens={lengths['p95']}, max_tokens={lengths['maximum']}"
        )


if __name__ == "__main__":
    main()
