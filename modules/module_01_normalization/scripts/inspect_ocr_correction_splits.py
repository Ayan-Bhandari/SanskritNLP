"""Inspect the pinned OCR correction splits and check target-text overlap."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import pyarrow.parquet as pq

MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from normalization import normalize_text


DATA_DIRECTORY = PROJECT_ROOT / "datasets" / "raw" / "sanskrit_ocr_post_correction"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "normalization" / "real_ocr" / "split_audit.json"
DATASET_ID = "acomquest/sanskrit-ocr-post-correction"
DATASET_REVISION = "35de91d045dfd8c9bd875b44f2bfda52f09f72ce"


def iter_parquet_rows(file_path):
    parquet_file = pq.ParquetFile(file_path)
    if parquet_file.schema_arrow.names != ["input_text", "target_text"]:
        raise ValueError(f"Unexpected schema in {file_path.name}.")
    for batch in parquet_file.iter_batches(batch_size=2_000):
        yield from batch.to_pylist()


def iter_csv_rows(file_path, delimiter):
    with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        if reader.fieldnames != ["input_text", "target_text"]:
            raise ValueError(f"Unexpected columns in {file_path.name}: {reader.fieldnames}")
        yield from reader


def normalized_hash(text):
    canonical_text = normalize_text(unicodedata.normalize("NFC", text))
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def audit_split(split_name, rows):
    summary = {
        "rows": 0,
        "empty_input_rows": 0,
        "empty_target_rows": 0,
        "invalid_pair_rows": 0,
        "input_not_nfc": 0,
        "target_not_nfc": 0,
        "input_changed_by_normalizer": 0,
        "target_changed_by_normalizer": 0,
        "input_with_devanagari_letters": 0,
        "target_with_devanagari_letters": 0,
        "input_chars": 0,
        "target_chars": 0,
        "input_max_chars": 0,
        "target_max_chars": 0,
    }
    target_hashes = set()
    duplicate_targets = 0

    for row in rows:
        input_text = row.get("input_text")
        target_text = row.get("target_text")
        summary["rows"] += 1

        if not isinstance(input_text, str) or not input_text.strip():
            summary["empty_input_rows"] += 1
        if not isinstance(target_text, str) or not target_text.strip():
            summary["empty_target_rows"] += 1
        if not isinstance(input_text, str) or not input_text.strip() or not isinstance(target_text, str) or not target_text.strip():
            summary["invalid_pair_rows"] += 1
            continue

        summary["input_chars"] += len(input_text)
        summary["target_chars"] += len(target_text)
        summary["input_max_chars"] = max(summary["input_max_chars"], len(input_text))
        summary["target_max_chars"] = max(summary["target_max_chars"], len(target_text))
        summary["input_not_nfc"] += not unicodedata.is_normalized("NFC", input_text)
        summary["target_not_nfc"] += not unicodedata.is_normalized("NFC", target_text)
        summary["input_changed_by_normalizer"] += normalize_text(input_text) != input_text
        summary["target_changed_by_normalizer"] += normalize_text(target_text) != target_text
        summary["input_with_devanagari_letters"] += any(
            unicodedata.category(character).startswith("L")
            and 0x0900 <= ord(character) <= 0x097F
            for character in input_text
        )
        summary["target_with_devanagari_letters"] += any(
            unicodedata.category(character).startswith("L")
            and 0x0900 <= ord(character) <= 0x097F
            for character in target_text
        )

        target_key = normalized_hash(target_text)
        if target_key in target_hashes:
            duplicate_targets += 1
        target_hashes.add(target_key)

    valid_rows = summary["rows"] - summary["invalid_pair_rows"]
    summary["valid_pairs"] = valid_rows
    summary["mean_input_chars"] = summary["input_chars"] / max(valid_rows, 1)
    summary["mean_target_chars"] = summary["target_chars"] / max(valid_rows, 1)
    summary["duplicate_targets_after_first"] = duplicate_targets
    return summary, target_hashes


def main():
    split_readers = {
        "train": iter_parquet_rows(DATA_DIRECTORY / "train.parquet"),
        "validation": iter_parquet_rows(DATA_DIRECTORY / "validation.parquet"),
        "test": iter_csv_rows(DATA_DIRECTORY / "test_devnagari.csv", ","),
        "ood_test": iter_csv_rows(DATA_DIRECTORY / "ood-test.csv", ";"),
    }
    summaries = {}
    target_hash_sets = {}
    for split_name, rows in split_readers.items():
        print(f"Inspecting {split_name}...")
        summaries[split_name], target_hash_sets[split_name] = audit_split(split_name, rows)
        print(f"  rows={summaries[split_name]['rows']:,}")

    overlap = {}
    split_names = list(target_hash_sets)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            key = f"{left_name}__{right_name}"
            overlap[key] = len(target_hash_sets[left_name].intersection(target_hash_sets[right_name]))

    report = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "declared_license": "MIT",
        "split_summaries": summaries,
        "normalized_target_overlap_between_splits": overlap,
        "target_duplicate_definition": "NFC plus the project's normalize_text whitespace rules",
        "comparison_notes": {
            "paper_counts": {"train": 208173, "validation": 5000, "test": 5000},
            "hugging_face_card_counts": {"train": 208138, "validation": 23126, "test": 34691, "ood_test": 500},
            "these_counts": "Counts read directly from the pinned local files.",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved split audit: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
