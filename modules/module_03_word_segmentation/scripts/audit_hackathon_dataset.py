"""Audit the Hackathon DCS segmentation CSV without changing the source."""

import argparse
import collections
import hashlib
import json
import platform
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = ["input", "output", "split", "DCS-ID"]
SPLITS = ("train", "dev", "test")
PAPER_URL = "https://aclanthology.org/2022.findings-emnlp.513/"
REPOSITORY_URL = (
    "https://github.com/Jivnesh/SanskritShala/tree/master/"
    "Neural_Modules/TransLIST/LREC-Data"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    series = pd.Series(values, dtype="int64")
    return {
        f"p{int(q * 100)}": float(series.quantile(q))
        for q in (0, 0.5, 0.9, 0.95, 0.99, 1)
    }


def value_overlap(left, right, column):
    left_values = set(left[column].dropna())
    right_values = set(right[column].dropna())
    shared = left_values & right_values
    return {"distinct_shared_values": len(shared), "examples": list(sorted(shared, key=str)[:5])}


def pair_overlap(left, right):
    left_values = set(zip(left["input"], left["output"]))
    right_values = set(zip(right["input"], right["output"]))
    return len(left_values & right_values)


def duplicate_summary(frame, key_columns):
    grouped = frame.groupby(key_columns, dropna=False, sort=False).size()
    repeated = grouped[grouped > 1]
    return {
        "distinct_groups": int(len(grouped)),
        "groups_repeated": int(len(repeated)),
        "rows_beyond_first": int((repeated - 1).sum()),
        "rows_in_repeated_groups": int(repeated.sum()),
    }


def unicode_summary(frame):
    counts = collections.Counter()
    text_by_column = {
        column: "\n".join(frame[column].dropna().astype(str).tolist())
        for column in ("input", "output")
    }
    per_column = {}
    for column, text in text_by_column.items():
        column_counts = collections.Counter(text)
        per_column[column] = column_counts
        counts.update(column_counts)

    def script_bucket(char):
        category = unicodedata.category(char)
        name = unicodedata.name(char, "")
        if char.isspace():
            return "whitespace"
        if name.startswith("LATIN "):
            return "latin"
        if name.startswith("DEVANAGARI "):
            return "devanagari"
        if category.startswith("M"):
            return "combining_or_spacing_mark"
        if category.startswith("P"):
            return "punctuation"
        if category.startswith("N"):
            return "number"
        if category.startswith("C"):
            return "control_or_format"
        return "other"

    buckets = collections.Counter()
    for char, count in counts.items():
        buckets[script_bucket(char)] += count
    return {
        "unique_characters_in_input_and_output": len(counts),
        "total_character_instances_in_input_and_output": sum(counts.values()),
        "character_instance_counts_by_category": dict(buckets),
        "top_characters": [
            {
                "character": char,
                "unicode_name": unicodedata.name(char, "UNNAMED"),
                "category": unicodedata.category(char),
                "count_input": per_column["input"][char],
                "count_output": per_column["output"][char],
                "count_total": count,
            }
            for char, count in counts.most_common(50)
        ],
        "unicode_nfc_rows_not_normalized": {
            column: int(
                frame[column].map(
                    lambda value: unicodedata.normalize("NFC", value) != value
                ).sum()
            )
            for column in ("input", "output")
        },
        "latin_letter_rows": {
            column: int(
                frame[column].map(
                    lambda text: any(
                        unicodedata.name(char, "").startswith("LATIN ")
                        for char in text
                    )
                ).sum()
            )
            for column in ("input", "output")
        },
        "devanagari_rows": {
            column: int(
                frame[column].map(
                    lambda text: any(
                        unicodedata.name(char, "").startswith("DEVANAGARI ")
                        for char in text
                    )
                ).sum()
            )
            for column in ("input", "output")
        },
    }


def text_profile(frame, column):
    values = frame[column]
    lengths = values.str.len()
    word_counts = values.str.split().str.len()
    return {
        "character_length": quantiles(lengths),
        "whitespace_word_count": quantiles(word_counts),
        "empty_or_whitespace_only_rows": int(values.str.strip().eq("").sum()),
        "leading_or_trailing_whitespace_rows": int(values.ne(values.str.strip()).sum()),
        "repeated_whitespace_rows": int(values.str.contains(r"\s{2,}", regex=True).sum()),
        "tab_rows": int(values.str.contains("\t", regex=False).sum()),
        "line_break_rows": int(values.str.contains(r"[\r\n]", regex=True).sum()),
        "non_ascii_rows": int(values.map(lambda text: any(ord(char) > 127 for char in text)).sum()),
    }


def run_audit(input_path, output_directory):
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    frame = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        dtype={column: "string" for column in EXPECTED_COLUMNS},
    )
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected columns {EXPECTED_COLUMNS}; found {list(frame.columns)}"
        )
    unknown_splits = sorted(set(frame["split"]) - set(SPLITS))

    exact_row_keys = EXPECTED_COLUMNS
    exact_pair_keys = ["input", "output"]
    per_split = {}
    for split in SPLITS:
        part = frame.loc[frame["split"] == split]
        per_split[split] = {
            "rows": len(part),
            "unique_inputs": int(part["input"].nunique()),
            "unique_input_output_pairs": int(part[exact_pair_keys].drop_duplicates().shape[0]),
            "unique_dcs_ids": int(part["DCS-ID"].nunique()),
            "exact_duplicate_records": duplicate_summary(part, exact_row_keys),
            "duplicate_inputs": duplicate_summary(part, ["input"]),
            "duplicate_input_output_pairs": duplicate_summary(part, exact_pair_keys),
            "duplicate_dcs_ids": duplicate_summary(part, ["DCS-ID"]),
            "input_profile": text_profile(part, "input"),
            "output_profile": text_profile(part, "output"),
            "examples": part[["input", "output", "DCS-ID"]].head(3).to_dict("records"),
        }

    input_output_counts = frame.groupby("input", sort=False)["output"].nunique()
    multi_output_inputs = input_output_counts[input_output_counts > 1].index
    multi_output_rows = frame.loc[frame["input"].isin(multi_output_inputs)]
    multi_output_examples = []
    for input_text, group in multi_output_rows.groupby("input", sort=False):
        outputs = group[["output", "split", "DCS-ID"]].drop_duplicates()
        multi_output_examples.append(
            {"input": input_text, "annotations": outputs.to_dict("records")}
        )
        if len(multi_output_examples) == 8:
            break

    dcs_input_counts = frame.groupby("DCS-ID", sort=False)["input"].nunique()
    pairs = {}
    for left_name, right_name in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left = frame.loc[frame["split"] == left_name]
        right = frame.loc[frame["split"] == right_name]
        pairs[f"{left_name}_vs_{right_name}"] = {
            "input_overlap": value_overlap(left, right, "input"),
            "input_output_pair_overlap_count": pair_overlap(left, right),
            "dcs_id_overlap": value_overlap(left, right, "DCS-ID"),
        }

    input_split_counts = frame.groupby("input", sort=False)["split"].nunique()
    id_split_counts = frame.groupby("DCS-ID", sort=False)["split"].nunique()
    pair_split_counts = frame.groupby(exact_pair_keys, sort=False)["split"].nunique()
    exact_across_splits = frame.duplicated(exact_row_keys, keep=False)

    input_compact = frame["input"].str.replace(r"\s+", "", regex=True)
    output_compact = frame["output"].str.replace(r"\s+", "", regex=True)
    whitespace_compact_equal = input_compact.eq(output_compact)
    row_characters = frame["input"].str.len() + frame["output"].str.len()
    char_difference_examples = []
    mismatch_indices = frame.index[~whitespace_compact_equal]
    for index in mismatch_indices[:12]:
        char_difference_examples.append(
            {
                "split": frame.at[index, "split"],
                "dcs_id": frame.at[index, "DCS-ID"],
                "input": frame.at[index, "input"],
                "output": frame.at[index, "output"],
                "input_without_whitespace": input_compact.at[index],
                "output_without_whitespace": output_compact.at[index],
            }
        )

    global_stats = {
        "rows": len(frame),
        "columns": [
            {"name": column, "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ],
        "split_values": frame["split"].value_counts(dropna=False).to_dict(),
        "unexpected_split_values": unknown_splits,
        "missing_values": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "empty_strings": {
            column: int(
                (frame[column].notna() & frame[column].str.strip().eq("")).sum()
            )
            for column in ("input", "output", "split", "DCS-ID")
        },
        "unique_inputs": int(frame["input"].nunique()),
        "unique_input_output_pairs": int(frame[exact_pair_keys].drop_duplicates().shape[0]),
        "unique_dcs_ids": int(frame["DCS-ID"].nunique()),
        "exact_duplicate_records": duplicate_summary(frame, exact_row_keys),
        "duplicate_inputs": duplicate_summary(frame, ["input"]),
        "duplicate_input_output_pairs": duplicate_summary(frame, exact_pair_keys),
        "duplicate_dcs_ids": duplicate_summary(frame, ["DCS-ID"]),
        "inputs_with_multiple_distinct_outputs": int(input_output_counts.gt(1).sum()),
        "rows_with_an_input_having_multiple_outputs": len(multi_output_rows),
        "dcs_ids_linked_to_multiple_distinct_inputs": int(dcs_input_counts.gt(1).sum()),
        "inputs_across_multiple_splits": {
            "distinct_inputs": int(input_split_counts.gt(1).sum()),
            "rows_involved": int(frame[frame["input"].isin(input_split_counts[input_split_counts.gt(1)].index)].shape[0]),
        },
        "dcs_ids_across_multiple_splits": {
            "distinct_ids": int(id_split_counts.gt(1).sum()),
            "rows_involved": int(frame[frame["DCS-ID"].isin(id_split_counts[id_split_counts.gt(1)].index)].shape[0]),
        },
        "input_output_pairs_across_multiple_splits": {
            "distinct_pairs": int(pair_split_counts.gt(1).sum()),
            "rows_involved": int(frame.set_index(exact_pair_keys).index.isin(pair_split_counts[pair_split_counts.gt(1)].index).sum()),
        },
        "exact_duplicate_rows_that_cross_splits": int(
            exact_across_splits.sum()
            and frame.loc[exact_across_splits].groupby(exact_row_keys, sort=False)["split"].nunique().gt(1).sum()
        ),
        "pairwise_split_overlaps": pairs,
        "input_output_character_relationship": {
            "rows_same_after_removing_all_whitespace": int(whitespace_compact_equal.sum()),
            "rows_different_after_removing_all_whitespace": int((~whitespace_compact_equal).sum()),
            "fraction_same_after_removing_all_whitespace": float(whitespace_compact_equal.mean()),
            "rows_with_more_whitespace_words_in_output": int(
                (frame["output"].str.split().str.len() > frame["input"].str.split().str.len()).sum()
            ),
            "rows_with_fewer_whitespace_words_in_output": int(
                (frame["output"].str.split().str.len() < frame["input"].str.split().str.len()).sum()
            ),
            "rows_with_equal_whitespace_word_count": int(
                (frame["output"].str.split().str.len() == frame["input"].str.split().str.len()).sum()
            ),
            "character_length_difference_output_minus_input": quantiles(
                frame["output"].str.len() - frame["input"].str.len()
            ),
            "changed_character_examples": char_difference_examples,
        },
        "multiple_output_examples": multi_output_examples,
        "unicode": unicode_summary(frame),
        "all_text_profiles": {
            "input": text_profile(frame, "input"),
            "output": text_profile(frame, "output"),
        },
        "representative_examples_by_split": {
            split: per_split[split]["examples"] for split in SPLITS
        },
    }

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_file_size_bytes": input_path.stat().st_size,
        "input_file_sha256": sha256_file(input_path),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "source_reference": REPOSITORY_URL,
        "research_reference": PAPER_URL,
        "statistics": global_stats,
        "per_split": per_split,
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "hackathon_dcs_audit.json"
    markdown_path = output_directory / "hackathon_dcs_audit.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report, json_path, markdown_path


def render_markdown(report):
    stats = report["statistics"]
    lines = [
        "# Hackathon DCS dataset audit",
        "",
        "## Scope and provenance",
        "",
        f"- Local input: `{report['input_file']}`",
        f"- File size: {report['input_file_size_bytes']:,} bytes",
        f"- SHA-256: `{report['input_file_sha256']}`",
        f"- Rows: {stats['rows']:,}",
        f"- Columns: {', '.join(item['name'] for item in stats['columns'])}",
        f"- Audit run (UTC): {report['audit_timestamp_utc']}",
        "- This script only reads the source CSV; it does not prepare or modify data.",
        "",
        "The local file's split counts can be compared with the TransLIST paper, which reports Hackathon train/dev/test sizes of 90,000 / 10,332 / 9,963. Matching counts support the file's benchmark identity but do not by themselves prove byte-for-byte identity with an upstream release.",
        "",
        "## Split distribution and missingness",
        "",
        "| Split | Rows | Unique inputs | Unique input/output pairs | Unique DCS IDs |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        part = report["per_split"][split]
        lines.append(
            f"| {split} | {part['rows']:,} | {part['unique_inputs']:,} | "
            f"{part['unique_input_output_pairs']:,} | {part['unique_dcs_ids']:,} |"
        )
    lines += [
        "",
        f"Missing values: `{stats['missing_values']}`. Empty or whitespace-only text fields: `{stats['empty_strings']}`.",
        "",
        "## Duplicate and ambiguity findings",
        "",
        f"- Exact duplicate records (all four fields): {stats['exact_duplicate_records']['rows_beyond_first']:,} rows beyond the first.",
        f"- Repeated inputs: {stats['duplicate_inputs']['groups_repeated']:,} input groups, {stats['duplicate_inputs']['rows_beyond_first']:,} extra rows.",
        f"- Repeated input/output pairs: {stats['duplicate_input_output_pairs']['groups_repeated']:,} groups, {stats['duplicate_input_output_pairs']['rows_beyond_first']:,} extra rows.",
        f"- Repeated DCS IDs: {stats['duplicate_dcs_ids']['groups_repeated']:,} IDs, {stats['duplicate_dcs_ids']['rows_beyond_first']:,} extra rows.",
        f"- Inputs with multiple distinct output annotations: {stats['inputs_with_multiple_distinct_outputs']:,} inputs across {stats['rows_with_an_input_having_multiple_outputs']:,} rows.",
        "",
        "Repeated rows and annotations are counted separately. Repetition alone is not treated as an error, and no record is removed by this audit.",
        "",
        "### Multiple-output examples",
        "",
    ]
    for example in stats["multiple_output_examples"]:
        lines.append(f"- Input: `{example['input']}`")
        for annotation in example["annotations"]:
            lines.append(
                f"  - Output: `{annotation['output']}`; split `{annotation['split']}`; DCS-ID `{annotation['DCS-ID']}`"
            )
    lines += ["", "## Cross-split overlap", ""]
    for pair_name, result in stats["pairwise_split_overlaps"].items():
        lines.append(
            f"- **{pair_name}:** {result['input_overlap']['distinct_shared_values']:,} shared inputs; "
            f"{result['input_output_pair_overlap_count']:,} shared exact input/output pairs; "
            f"{result['dcs_id_overlap']['distinct_shared_values']:,} shared DCS IDs."
        )
    lines += [
        "",
        f"Across all splits, {stats['inputs_across_multiple_splits']['distinct_inputs']:,} distinct inputs occur in more than one split ({stats['inputs_across_multiple_splits']['rows_involved']:,} rows), and {stats['dcs_ids_across_multiple_splits']['distinct_ids']:,} DCS IDs cross splits.",
        "",
        "## Text and annotation format",
        "",
        f"Rows whose input and output become identical after deleting all whitespace: {stats['input_output_character_relationship']['rows_same_after_removing_all_whitespace']:,} / {stats['rows']:,}.",
        f"Rows with character-sequence differences beyond whitespace: {stats['input_output_character_relationship']['rows_different_after_removing_all_whitespace']:,}.",
        "",
        "If non-whitespace characters change, a model that only predicts spaces at input character positions cannot represent every gold output. A character-level boundary tagger may still be a baseline on the aligned subset, but the main formulation should support sequence edits or generate the segmented output. This conclusion should be revisited after reviewing the detailed examples below.",
        "",
        "### Character-change examples",
        "",
    ]
    for example in stats["input_output_character_relationship"]["changed_character_examples"]:
        lines.append(
            f"- `{example['input']}` → `{example['output']}` ({example['split']}, DCS-ID {example['dcs_id']})"
        )
    lines += ["", "## Transliteration and length profile", ""]
    lines.append(
        "- Unicode normalization differences (NFC): "
        f"{stats['unicode']['unicode_nfc_rows_not_normalized']}"
    )
    lines.append(
        "- Rows containing Latin letters: "
        f"{stats['unicode']['latin_letter_rows']}; rows containing Devanagari: "
        f"{stats['unicode']['devanagari_rows']}."
    )
    lines.append(
        f"- Unique characters in input and output combined: {stats['unicode']['unique_characters_in_input_and_output']:,}."
    )
    lines.append("")
    lines.append("Input/output length summaries are in the JSON report under `all_text_profiles`; top Unicode characters and categories are under `unicode`.")
    lines += [
        "",
        "## Recommendations before preparation or training",
        "",
        "1. Preserve the raw CSV and original benchmark partitions unchanged.",
        "2. Keep exact duplicate rows in an audit ledger; deduplicate only a derived training view if their repeated weighting is not intended.",
        "3. Preserve repeated inputs with alternative outputs until their linguistic validity is reviewed. For exact sentence match, evaluate against all accepted outputs or define a reviewed canonical annotation policy.",
        "4. Report benchmark-compatible results on original dev/test splits, but also create a separate leakage-controlled evaluation view if cross-split input overlap is nonzero. Keep test as the highest-priority partition and quarantine lower-priority copies of test inputs; document row counts and do not silently reshuffle.",
        "5. Do not use a boundary-only label formulation as the sole model if outputs change non-whitespace characters. Compare a boundary tagger baseline on alignable rows with a sequence-to-sequence/edit-capable model on the full set.",
        "6. Treat `DCS-ID` as a grouping/leakage field if repeated IDs are confirmed; do not use it as model input.",
        "7. Do not claim direct TransLIST comparability from matching split sizes alone; record preprocessing, candidate resources, and word-level metrics separately.",
        "",
        "## Repository compatibility and constraints",
        "",
        "Modules 1 and 2 were inspected and left unchanged. Module 1 provides NFC/whitespace normalization for Unicode Sanskrit text, but this Hackathon data is Roman transliteration; avoid applying new normalization rules until the diacritic conventions are documented. Module 2's BPE tokenization does not recover Sanskrit word boundaries and is not a replacement for this task.",
        "",
        "No cleaning, split regeneration, dataset preparation, or model training was performed. The next step requires the user's review and approval.",
        "",
        "## Research references",
        "",
        f"- TransLIST paper (correct ACL Anthology identifier): {PAPER_URL}",
        f"- Authors' TransLIST data directory: {REPOSITORY_URL}",
        "- The context prompt's `.300` ACL URL points to an unrelated FSAM paper; TransLIST is `.513`.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the user-provided Hackathon_dcs.csv")
    parser.add_argument(
        "--output-dir",
        default="experiments/module_03_segmentation",
        help="Local output directory for audit JSON and Markdown",
    )
    args = parser.parse_args()
    report, json_path, markdown_path = run_audit(args.input, args.output_dir)
    stats = report["statistics"]
    print(f"Rows: {stats['rows']:,}; splits: {stats['split_values']}")
    print(
        "Duplicates: "
        f"{stats['exact_duplicate_records']['rows_beyond_first']:,} exact rows; "
        f"{stats['inputs_with_multiple_distinct_outputs']:,} inputs have multiple outputs"
    )
    print(f"Cross-split input overlaps: {stats['inputs_across_multiple_splits']['distinct_inputs']:,}")
    print(f"Character-changing targets: {stats['input_output_character_relationship']['rows_different_after_removing_all_whitespace']:,}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {markdown_path}")


if __name__ == "__main__":
    main()
