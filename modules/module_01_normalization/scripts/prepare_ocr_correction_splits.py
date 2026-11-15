"""Create canonical, target-disjoint OCR correction splits."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from normalization import normalize_text


RAW_DIRECTORY = PROJECT_ROOT / "datasets" / "raw" / "sanskrit_ocr_post_correction"
OUTPUT_DIRECTORY = PROJECT_ROOT / "datasets" / "splits" / "ocr_target_disjoint"
DATASET_REVISION = "35de91d045dfd8c9bd875b44f2bfda52f09f72ce"


def target_key(text):
    canonical = normalize_text(unicodedata.normalize("NFC", text))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_parquet_split(file_path, split_name):
    rows = []
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=2_000):
        for row_number, row in enumerate(batch.to_pylist()):
            rows.append(make_row(row, split_name, len(rows) + 1))
    return rows


def read_csv_split(file_path, split_name, delimiter):
    rows = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        if reader.fieldnames != ["input_text", "target_text"]:
            raise ValueError(f"Unexpected columns in {file_path.name}: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=1):
            rows.append(make_row(row, split_name, row_number))
    return rows


def make_row(row, split_name, row_number):
    raw_input = row.get("input_text")
    raw_target = row.get("target_text")
    if not isinstance(raw_input, str) or not raw_input.strip():
        raise ValueError(f"Empty input in {split_name} row {row_number}.")
    if not isinstance(raw_target, str) or not raw_target.strip():
        raise ValueError(f"Empty target in {split_name} row {row_number}.")

    canonical_input = normalize_text(unicodedata.normalize("NFC", raw_input))
    canonical_target = normalize_text(unicodedata.normalize("NFC", raw_target))
    return {
        "source_split": split_name,
        "row_number": row_number,
        "raw_input_text": raw_input,
        "raw_target_text": raw_target,
        "input_text": canonical_input,
        "target_text": canonical_target,
        "target_key": target_key(canonical_target),
    }


def filter_by_targets(rows, excluded_target_keys):
    kept = []
    removed = 0
    for row in rows:
        if row["target_key"] in excluded_target_keys:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def write_split(split_name, rows):
    output_path = OUTPUT_DIRECTORY / f"{split_name}.parquet"
    output_rows = [
        {key: value for key, value in row.items() if key != "target_key"}
        for row in rows
    ]
    table = pa.Table.from_pylist(output_rows)
    pq.write_table(table, output_path, compression="zstd")
    return output_path


def main():
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIRECTORY}")

    source_splits = {
        "train": read_parquet_split(RAW_DIRECTORY / "train.parquet", "train"),
        "validation": read_parquet_split(RAW_DIRECTORY / "validation.parquet", "validation"),
        "test": read_csv_split(RAW_DIRECTORY / "test_devnagari.csv", "test", ","),
        "ood_test": read_csv_split(RAW_DIRECTORY / "ood-test.csv", "ood_test", ";"),
    }

    # Preserve OOD first, then test, then validation. Training never sees a
    # canonical target used by any evaluation split.
    ood_rows = source_splits["ood_test"]
    ood_targets = {row["target_key"] for row in ood_rows}

    test_rows, test_removed_for_ood = filter_by_targets(source_splits["test"], ood_targets)
    test_targets = {row["target_key"] for row in test_rows}

    validation_exclusions = ood_targets | test_targets
    validation_rows, validation_removed = filter_by_targets(
        source_splits["validation"], validation_exclusions
    )
    validation_targets = {row["target_key"] for row in validation_rows}

    training_exclusions = ood_targets | test_targets | validation_targets
    train_rows, train_removed = filter_by_targets(
        source_splits["train"], training_exclusions
    )

    prepared_splits = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
        "ood_test": ood_rows,
    }
    target_sets = {
        name: {row["target_key"] for row in rows}
        for name, rows in prepared_splits.items()
    }

    overlaps = {}
    names = list(target_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlaps[f"{left}__{right}"] = len(target_sets[left] & target_sets[right])
    if any(overlaps.values()):
        raise ValueError(f"Target overlap remains after preparation: {overlaps}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    written_files = {}
    for split_name, rows in prepared_splits.items():
        path = write_split(split_name, rows)
        written_files[split_name] = str(path.relative_to(PROJECT_ROOT))

    report = {
        "dataset_id": "acomquest/sanskrit-ocr-post-correction",
        "dataset_revision": DATASET_REVISION,
        "target_canonicalization": "NFC plus src.normalization.normalize_text",
        "split_priority": ["ood_test", "test", "validation", "train"],
        "source_counts": {name: len(rows) for name, rows in source_splits.items()},
        "prepared_counts": {name: len(rows) for name, rows in prepared_splits.items()},
        "removed_for_target_overlap": {
            "train": train_removed,
            "validation": validation_removed,
            "test": test_removed_for_ood,
            "ood_test": 0,
        },
        "unique_target_counts": {name: len(keys) for name, keys in target_sets.items()},
        "prepared_cross_split_target_overlap": overlaps,
        "retained_raw_text_columns": True,
        "files": written_files,
    }
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
