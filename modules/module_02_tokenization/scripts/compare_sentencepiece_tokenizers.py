"""Fit small CPU SentencePiece models and compare them on held-out texts."""

import json
import math
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
INPUT_PATH = PROJECT_ROOT / "datasets" / "splits" / "sansar_source_aware_group_splits.parquet"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "tokenization" / "sentencepiece"
REPORT_PATH = PROJECT_ROOT / "experiments" / "tokenization" / "module2_sentencepiece_comparison.json"
SPLIT_NAMES = ("train", "validation", "test")
VOCABULARY_SIZES = (512, 1024)
MODEL_TYPES = ("bpe", "unigram")
LINE_BREAK_TOKEN = "<SANSKRIT_NLP_LINEBREAK>"
TRAINER_THREADS = 4


def percentile(sorted_values, fraction):
    """Return a nearest-rank percentile from sorted values."""
    if not sorted_values:
        return 0
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def read_splits():
    """Read the small prepared split in Arrow batches, grouped by partition."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing input split: {INPUT_PATH}. Run Module 1 split preparation first."
        )

    parquet_file = pq.ParquetFile(INPUT_PATH)
    required_columns = {"split", "normalized_text"}
    actual_columns = set(parquet_file.schema_arrow.names)
    missing_columns = required_columns - actual_columns
    if missing_columns:
        raise ValueError(f"Input split is missing columns: {sorted(missing_columns)}")

    texts_by_split = {split_name: [] for split_name in SPLIT_NAMES}
    for batch in parquet_file.iter_batches(
        columns=["split", "normalized_text"],
        batch_size=512,
    ):
        for row in batch.to_pylist():
            split_name = row["split"]
            text = row["normalized_text"]
            if split_name not in texts_by_split:
                raise ValueError(f"Unexpected split label: {split_name!r}")
            if not isinstance(text, str):
                raise ValueError(f"Non-string normalized text in {split_name} split")
            if LINE_BREAK_TOKEN in text:
                raise ValueError(
                    "The line-break marker already occurs in source text; "
                    "choose a different marker before training."
                )
            texts_by_split[split_name].append(text)

    return parquet_file, texts_by_split


def sentence_iterator(training_texts):
    """Yield one record per training sentence with line breaks represented."""
    for text in training_texts:
        yield text.replace("\n", LINE_BREAK_TOKEN)


def train_model(training_texts, model_type, vocabulary_size, max_sentence_length):
    """Train one SentencePiece model using training rows only."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_prefix = OUTPUT_DIRECTORY / f"{model_type}_vocab{vocabulary_size}"
    started = time.perf_counter()

    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator(training_texts),
        model_prefix=str(model_prefix),
        model_type=model_type,
        vocab_size=vocabulary_size,
        hard_vocab_limit=False,
        character_coverage=1.0,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        split_by_whitespace=True,
        user_defined_symbols=[LINE_BREAK_TOKEN],
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        max_sentence_length=max_sentence_length,
        num_threads=TRAINER_THREADS,
        minloglevel=2,
    )

    elapsed = time.perf_counter() - started
    processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")
    return processor, model_prefix, elapsed


def encode_preserving_line_breaks(processor, text):
    """Encode normalized text with line breaks represented as one special unit."""
    model_text = text.replace("\n", LINE_BREAK_TOKEN)
    token_ids = processor.encode(model_text, out_type=int)
    decoded = processor.decode(token_ids)
    reconstructed = decoded.replace(LINE_BREAK_TOKEN, "\n")
    return token_ids, reconstructed


def summarize_partition(processor, texts):
    token_lengths = []
    unknown_token_instances = 0
    exact_round_trip_rows = 0
    line_breaks = 0
    subwords_per_whitespace_token = []

    for text in texts:
        token_ids, reconstructed = encode_preserving_line_breaks(processor, text)
        token_lengths.append(len(token_ids))
        unknown_token_instances += sum(
            token_id == processor.unk_id() for token_id in token_ids
        )
        exact_round_trip_rows += reconstructed == text
        line_breaks += text.count("\n")
        whitespace_token_count = len(text.split())
        if whitespace_token_count:
            subwords_per_whitespace_token.append(
                len(token_ids) / whitespace_token_count
            )

    ordered_lengths = sorted(token_lengths)
    total_tokens = sum(token_lengths)
    return {
        "rows": len(texts),
        "actual_vocabulary_size": processor.get_piece_size(),
        "total_subword_tokens": total_tokens,
        "mean_subword_tokens_per_row": round(
            statistics.mean(token_lengths), 2
        ) if token_lengths else 0,
        "median_subword_tokens_per_row": statistics.median(token_lengths) if token_lengths else 0,
        "p90_subword_tokens_per_row": percentile(ordered_lengths, 0.90),
        "p95_subword_tokens_per_row": percentile(ordered_lengths, 0.95),
        "maximum_subword_tokens_per_row": max(token_lengths, default=0),
        "mean_subwords_per_whitespace_token": round(
            statistics.mean(subwords_per_whitespace_token), 3
        ) if subwords_per_whitespace_token else 0,
        "unknown_token_instances": unknown_token_instances,
        "exact_round_trip_rows": exact_round_trip_rows,
        "line_break_markers_encoded": line_breaks,
    }


def main():
    parquet_file, texts_by_split = read_splits()
    training_texts = texts_by_split["train"]
    if not training_texts:
        raise ValueError("The train split is empty; cannot fit a tokenizer.")

    maximum_input_bytes = max(
        len(text.replace("\n", LINE_BREAK_TOKEN).encode("utf-8"))
        for text in training_texts
    )
    max_sentence_length = maximum_input_bytes + 1_024

    model_results = {}
    for model_type in MODEL_TYPES:
        for vocabulary_size in VOCABULARY_SIZES:
            model_key = f"{model_type}_vocab{vocabulary_size}"
            processor, model_prefix, training_seconds = train_model(
                training_texts,
                model_type,
                vocabulary_size,
                max_sentence_length,
            )

            split_results = {
                split_name: summarize_partition(processor, texts)
                for split_name, texts in texts_by_split.items()
            }
            model_results[model_key] = {
                "model_type": model_type,
                "requested_vocabulary_size": vocabulary_size,
                "actual_vocabulary_size": processor.get_piece_size(),
                "model_file": str(
                    (model_prefix.with_suffix(".model")).relative_to(PROJECT_ROOT)
                ),
                "vocabulary_file": str(
                    (model_prefix.with_suffix(".vocab")).relative_to(PROJECT_ROOT)
                ),
                "training_seconds": round(training_seconds, 3),
                "splits": split_results,
            }

            print(
                f"{model_key}: vocab={processor.get_piece_size()}, "
                f"val_mean={split_results['validation']['mean_subword_tokens_per_row']}, "
                f"test_mean={split_results['test']['mean_subword_tokens_per_row']}, "
                f"round_trip={split_results['test']['exact_round_trip_rows']}/"
                f"{split_results['test']['rows']}"
            )

    report = {
        "input_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "rows_in_file": parquet_file.metadata.num_rows,
        "sentencepiece_version": spm.__version__,
        "training_rows": len(training_texts),
        "training_policy": (
            "Fit each model using train rows only; validation and test are "
            "used only for reporting."
        ),
        "line_break_policy": (
            f"Replace LF with user-defined symbol {LINE_BREAK_TOKEN!r} during "
            "training/encoding, then restore LF after decoding."
        ),
        "normalization_policy": (
            "SentencePiece identity normalization; no extra whitespace collapse; "
            "no dummy prefix."
        ),
        "max_sentence_length_bytes": max_sentence_length,
        "trainer_threads": TRAINER_THREADS,
        "models": model_results,
        "limitations": [
            "This is a tokenizer-coverage and sequence-length comparison, not a downstream language-model quality evaluation.",
            "The work-group sample is small and training text has a long sequence-length tail.",
            "Exact round-trip counts include the explicit line-break marker wrapper.",
            "Whitespace boundaries do not recover Sandhi-hidden word boundaries.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Aggregate report: {REPORT_PATH}")
    print(f"Local model artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
