"""Evaluate the fixed text normalizer on a held-out Sanskrit OCR set."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

from rapidfuzz.distance import Levenshtein


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from normalization import normalize_text


INPUT_PATH = PROJECT_ROOT / "datasets" / "raw" / "sanskrit_ocr_post_correction" / "ood-test.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "normalization" / "real_ocr"
METRICS_PATH = OUTPUT_DIRECTORY / "ood_test_baseline_metrics.json"
PAIR_METRICS_PATH = OUTPUT_DIRECTORY / "ood_test_pair_metrics.csv"
DATASET_REVISION = "35de91d045dfd8c9bd875b44f2bfda52f09f72ce"
DATASET_ID = "acomquest/sanskrit-ocr-post-correction"


def nfc(text):
    """Normalize comparison strings to NFC without changing other content."""
    return unicodedata.normalize("NFC", text)


def error_distance(prediction, reference):
    return Levenshtein.distance(nfc(prediction), nfc(reference))


def cer_from_distance(distance, reference):
    reference_length = len(nfc(reference))
    if reference_length == 0:
        return 0.0 if distance == 0 else 1.0
    return distance / reference_length


def evaluate_rows(rows):
    """Compare identity and the already-frozen deterministic normalizer."""
    totals = {
        "reference_codepoints": 0,
        "identity_edit_distance": 0,
        "normalizer_edit_distance": 0,
        "identity_exact_matches": 0,
        "normalizer_exact_matches": 0,
        "rows_changed_by_normalizer": 0,
        "rows_cer_improved": 0,
        "rows_cer_worsened": 0,
        "rows_cer_unchanged": 0,
    }
    pair_rows = []

    for row_number, row in enumerate(rows, start=1):
        input_text = row["input_text"]
        target_text = row["target_text"]
        if not isinstance(input_text, str) or not isinstance(target_text, str):
            raise ValueError(f"Row {row_number} has a non-string input or target.")
        if not input_text.strip() or not target_text.strip():
            raise ValueError(f"Row {row_number} has an empty input or target.")

        normalized_input = normalize_text(input_text)
        normalized_target = nfc(target_text)
        identity_distance = error_distance(input_text, target_text)
        normalizer_distance = error_distance(normalized_input, target_text)
        identity_cer = cer_from_distance(identity_distance, target_text)
        normalizer_cer = cer_from_distance(normalizer_distance, target_text)

        totals["reference_codepoints"] += len(normalized_target)
        totals["identity_edit_distance"] += identity_distance
        totals["normalizer_edit_distance"] += normalizer_distance
        totals["identity_exact_matches"] += nfc(input_text) == normalized_target
        totals["normalizer_exact_matches"] += nfc(normalized_input) == normalized_target
        totals["rows_changed_by_normalizer"] += normalized_input != input_text

        if normalizer_cer < identity_cer:
            totals["rows_cer_improved"] += 1
        elif normalizer_cer > identity_cer:
            totals["rows_cer_worsened"] += 1
        else:
            totals["rows_cer_unchanged"] += 1

        pair_rows.append({
            "row_number": row_number,
            "reference_codepoints": len(normalized_target),
            "identity_edit_distance": identity_distance,
            "normalizer_edit_distance": normalizer_distance,
            "identity_cer": identity_cer,
            "normalizer_cer": normalizer_cer,
            "identity_exact_match": nfc(input_text) == normalized_target,
            "normalizer_exact_match": nfc(normalized_input) == normalized_target,
        })

    row_count = len(rows)
    if row_count == 0:
        raise ValueError("No OCR examples were loaded.")

    metrics = {
        "examples": row_count,
        "identity_micro_cer": totals["identity_edit_distance"] / max(totals["reference_codepoints"], 1),
        "normalizer_micro_cer": totals["normalizer_edit_distance"] / max(totals["reference_codepoints"], 1),
        "identity_mean_row_cer": sum(row["identity_cer"] for row in pair_rows) / row_count,
        "normalizer_mean_row_cer": sum(row["normalizer_cer"] for row in pair_rows) / row_count,
        "identity_exact_match_rate": totals["identity_exact_matches"] / row_count,
        "normalizer_exact_match_rate": totals["normalizer_exact_matches"] / row_count,
        "rows_changed_by_normalizer": totals["rows_changed_by_normalizer"],
        "rows_cer_improved": totals["rows_cer_improved"],
        "rows_cer_worsened": totals["rows_cer_worsened"],
        "rows_cer_unchanged": totals["rows_cer_unchanged"],
    }
    return metrics, pair_rows


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"OCR input file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter=";")
        if reader.fieldnames != ["input_text", "target_text"]:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        rows = list(reader)

    metrics, pair_rows = evaluate_rows(rows)
    file_sha256 = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    report = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "ood_test",
        "declared_license": "MIT",
        "source_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "source_file_sha256": file_sha256,
        "normalizer": "src.normalization.text_normalizer.normalize_text",
        "metrics": metrics,
        "method": (
            "Identity returns OCR input unchanged. Rule baseline applies the "
            "existing deterministic normalizer. CER compares NFC prediction "
            "and NFC target; no rules were tuned on this held-out split."
        ),
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with PAIR_METRICS_PATH.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"Examples: {metrics['examples']:,}")
    print(f"Identity micro-CER: {metrics['identity_micro_cer']:.6f}")
    print(f"Normalizer micro-CER: {metrics['normalizer_micro_cer']:.6f}")
    print(f"Identity exact match: {metrics['identity_exact_match_rate']:.3%}")
    print(f"Normalizer exact match: {metrics['normalizer_exact_match_rate']:.3%}")
    print(f"Rows changed by normalizer: {metrics['rows_changed_by_normalizer']:,}")
    print(f"Saved report: {METRICS_PATH}")
    print(f"Saved per-row metrics: {PAIR_METRICS_PATH}")


if __name__ == "__main__":
    main()
