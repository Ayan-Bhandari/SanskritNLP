"""Compare lossless whitespace and Unicode-code-point tokenizers."""

import json
import math
import statistics
import time
from pathlib import Path
import re

import pyarrow.parquet as pq


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
INPUT_PATH = PROJECT_ROOT / "datasets" / "splits" / "sansar_source_aware_group_splits.parquet"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "tokenization" / "module2_baseline_tokenizers.json"
SPLIT_NAMES = ("train", "validation", "test")
WHITESPACE_PATTERN = re.compile(r"\S+")


def percentile(sorted_values, fraction):
    """Return the nearest-rank percentile from sorted values."""
    if not sorted_values:
        return 0
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def tokenize_whitespace(text):
    """Return non-whitespace tokens and the exact separators around them."""
    tokens = []
    separators = []
    previous_end = 0

    for match in WHITESPACE_PATTERN.finditer(text):
        separators.append(text[previous_end:match.start()])
        tokens.append(match.group())
        previous_end = match.end()

    trailing_whitespace = text[previous_end:]
    return tokens, separators, trailing_whitespace


def detokenize_whitespace(tokens, separators, trailing_whitespace):
    """Reconstruct the source string from tokens and saved whitespace."""
    pieces = []
    for token, separator in zip(tokens, separators):
        pieces.append(separator)
        pieces.append(token)
    pieces.append(trailing_whitespace)
    return "".join(pieces)


def tokenize_codepoints(text):
    """Return one Unicode code point per token, including whitespace."""
    return list(text)


def summarize_lengths(token_lists):
    lengths = sorted(len(tokens) for tokens in token_lists)
    return {
        "total_tokens": sum(lengths),
        "mean_tokens_per_row": round(statistics.mean(lengths), 2) if lengths else 0,
        "median_tokens_per_row": statistics.median(lengths) if lengths else 0,
        "p90_tokens_per_row": percentile(lengths, 0.90),
        "p95_tokens_per_row": percentile(lengths, 0.95),
        "maximum_tokens_per_row": max(lengths, default=0),
    }


def summarize_oov(token_lists, training_vocabulary):
    unknown_instances = 0
    total_instances = 0
    unknown_types = set()

    for tokens in token_lists:
        for token in tokens:
            total_instances += 1
            if token not in training_vocabulary:
                unknown_instances += 1
                unknown_types.add(token)

    return {
        "unknown_token_instances": unknown_instances,
        "total_token_instances": total_instances,
        "unknown_instance_rate": round(
            unknown_instances / max(total_instances, 1), 6
        ),
        "unseen_token_types": len(unknown_types),
    }


def read_rows_by_split():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing input split: {INPUT_PATH}. Run the Module 1 split "
            "preparation first; this script does not download or modify data."
        )

    parquet_file = pq.ParquetFile(INPUT_PATH)
    required_columns = {"split", "normalized_text"}
    actual_columns = set(parquet_file.schema_arrow.names)
    missing_columns = required_columns - actual_columns
    if missing_columns:
        raise ValueError(f"Input split is missing columns: {sorted(missing_columns)}")

    rows_by_split = {split_name: [] for split_name in SPLIT_NAMES}
    for batch in parquet_file.iter_batches(
        columns=["split", "normalized_text"],
        batch_size=512,
    ):
        for row in batch.to_pylist():
            split_name = row["split"]
            text = row["normalized_text"]
            if split_name not in rows_by_split:
                raise ValueError(f"Unexpected split label: {split_name!r}")
            if not isinstance(text, str):
                raise ValueError(f"Non-string normalized_text in {split_name} split")
            rows_by_split[split_name].append(text)

    return parquet_file, rows_by_split


def main():
    parquet_file, texts_by_split = read_rows_by_split()
    tokenizers = {
        "whitespace": tokenize_whitespace,
        "unicode_codepoint": tokenize_codepoints,
    }

    elapsed_seconds = {}
    tokens_by_split = {name: {} for name in SPLIT_NAMES}
    whitespace_gaps_by_split = {name: [] for name in SPLIT_NAMES}

    for tokenizer_name, tokenizer in tokenizers.items():
        started = time.perf_counter()
        for split_name, texts in texts_by_split.items():
            for text in texts:
                if tokenizer_name == "whitespace":
                    tokens, separators, trailing = tokenizer(text)
                    reconstructed = detokenize_whitespace(tokens, separators, trailing)
                    whitespace_gaps_by_split[split_name].append(
                        reconstructed == text
                    )
                else:
                    tokens = tokenizer(text)
                    reconstructed = "".join(tokens)
                    if reconstructed != text:
                        raise ValueError("Code-point detokenization changed the input")
                tokens_by_split[split_name].setdefault(tokenizer_name, []).append(tokens)
        elapsed_seconds[tokenizer_name] = round(time.perf_counter() - started, 4)

    training_vocabularies = {
        tokenizer_name: {
            token
            for tokens in tokens_by_split["train"][tokenizer_name]
            for token in tokens
        }
        for tokenizer_name in tokenizers
    }

    split_results = {}
    for split_name in SPLIT_NAMES:
        split_results[split_name] = {}
        for tokenizer_name in tokenizers:
            token_lists = tokens_by_split[split_name][tokenizer_name]
            result = summarize_lengths(token_lists)
            result["exact_round_trip_rows"] = sum(
                whitespace_gaps_by_split[split_name]
            ) if tokenizer_name == "whitespace" else len(token_lists)
            result["rows"] = len(token_lists)
            result["training_vocabulary_types"] = len(
                training_vocabularies[tokenizer_name]
            )
            if split_name != "train":
                result["out_of_vocabulary"] = summarize_oov(
                    token_lists,
                    training_vocabularies[tokenizer_name],
                )
            split_results[split_name][tokenizer_name] = result

    whitespace_tokens = [
        token
        for token_lists in tokens_by_split["train"]["whitespace"]
        for token in token_lists
    ]
    danda_token_instances = sum(
        "।" in token or "॥" in token for token in whitespace_tokens
    )

    report = {
        "input_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "rows_in_file": parquet_file.metadata.num_rows,
        "fitting_policy": "Both vocabularies are built from train only.",
        "tokenizer_definitions": {
            "whitespace": "Non-whitespace spans; exact whitespace separators are kept for reversible reconstruction.",
            "unicode_codepoint": "One Python Unicode code point per token, including spaces and line breaks.",
        },
        "elapsed_seconds": elapsed_seconds,
        "whitespace_train_tokens_containing_danda": danda_token_instances,
        "splits": split_results,
        "limitations": [
            "The whitespace tokenizer follows source spacing and does not recover Sandhi-hidden word boundaries.",
            "The code-point tokenizer may split Devanagari combining sequences; it is a diagnostic baseline, not the final character design.",
            "Round-trip preservation is not a measure of Sanskrit linguistic accuracy.",
            "Validation and test are reported only; their text was not used to build vocabularies.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Rows compared: {report['rows_in_file']:,}")
    print(f"Aggregate results: {OUTPUT_PATH}")
    for split_name, tokenizers_result in split_results.items():
        print(f"{split_name}:")
        for tokenizer_name, result in tokenizers_result.items():
            print(
                f"  {tokenizer_name}: rows={result['rows']:,}, "
                f"mean_units={result['mean_tokens_per_row']}, "
                f"vocabulary={result['training_vocabulary_types']}, "
                f"exact_round_trip={result['exact_round_trip_rows']:,}"
            )


if __name__ == "__main__":
    main()
