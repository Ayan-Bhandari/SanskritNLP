"""Evaluate sentence segmentation by decoding existing whitespace chunks separately.

This dev-first experiment preserves input whitespace boundaries and asks the
character model to segment each whitespace-delimited span independently.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from rapidfuzz.distance import Levenshtein

from train_character_transformer import (
    DEFAULT_DATASET, MAX_CHARACTER_EDIT_RATIO, MODEL_DIR, CharacterTransformer,
    EOS, PAD, greedy_decode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "module_03_segmentation" / "error_analysis"


def make_source_batches(chunks, token_to_id, batch_size):
    ordered = sorted(range(len(chunks)), key=lambda index: len(chunks[index]["text"]))
    for start in range(0, len(ordered), batch_size):
        selected = [chunks[i] for i in ordered[start:start + batch_size]]
        sequences = [
            [token_to_id.get(char, token_to_id["<unk>"]) for char in item["text"]] + [token_to_id[EOS]]
            for item in selected
        ]
        source = torch.full((len(selected), max(map(len, sequences))), token_to_id[PAD], dtype=torch.long)
        for row_index, sequence in enumerate(sequences):
            source[row_index, :len(sequence)] = torch.tensor(sequence)
        yield source, selected


def best_reference_scores(prediction, references):
    best = None
    for reference in references:
        pred_words, gold_words = prediction.split(), reference.split()
        overlap = sum((Counter(pred_words) & Counter(gold_words)).values())
        precision = overlap / len(pred_words) if pred_words else float(not gold_words)
        recall = overlap / len(gold_words) if gold_words else float(not pred_words)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = {"precision": precision, "recall": recall, "f1": f1,
                     "exact": prediction == reference, "reference": reference}
        if best is None or (candidate["f1"], candidate["exact"]) > (best["f1"], best["exact"]):
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "best_model.pt")
    parser.add_argument("--split", choices=("dev",), default="dev",
                        help="This candidate is deliberately tuned on dev only.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokens = checkpoint["tokens"]
    token_to_id = {token: i for i, token in enumerate(tokens)}
    max_length = checkpoint["config"]["max_length"]
    model = CharacterTransformer(len(tokens), max_length).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frame = pd.read_csv(args.input, dtype="string")
    rows = frame[frame["split"].eq(args.split)].drop_duplicates(["input", "output"])
    references = defaultdict(list)
    for row in rows[["input", "output"]].itertuples(index=False):
        if row.output not in references[row.input]:
            references[row.input].append(row.output)
    inputs = list(references)
    chunk_records = []
    for row_index, source in enumerate(inputs):
        for chunk_index, chunk in enumerate(source.split()):
            chunk_records.append({"row_index": row_index, "chunk_index": chunk_index, "text": chunk})

    decoded = {}
    for batch, selected in make_source_batches(chunk_records, token_to_id, args.batch_size):
        predictions = greedy_decode(model, batch.to(device), token_to_id, tokens, max_length)
        for item, prediction in zip(selected, predictions):
            source_chars, predicted_chars = item["text"], "".join(prediction.split())
            ratio = Levenshtein.distance(source_chars, predicted_chars) / max(len(source_chars), 1)
            fallback = ratio > MAX_CHARACTER_EDIT_RATIO
            decoded.setdefault(item["row_index"], []).append(
                (item["chunk_index"], item["text"] if fallback else prediction, fallback)
            )

    exact = 0
    totals = Counter()
    macro = Counter()
    output_rows = []
    for row_index, source in enumerate(inputs):
        predicted_chunks = sorted(decoded[row_index])
        prediction = " ".join(item[1] for item in predicted_chunks)
        score = best_reference_scores(prediction, references[source])
        exact += int(score["exact"])
        macro.update({key: score[key] for key in ("precision", "recall", "f1")})
        totals["fallback_chunks"] += sum(item[2] for item in predicted_chunks)
        totals["chunks"] += len(predicted_chunks)
        totals["fallback_sentences"] += any(item[2] for item in predicted_chunks)
        output_rows.append({"input": source, "prediction": prediction,
                            "references_json": json.dumps(references[source], ensure_ascii=False),
                            "exact_match_any_reference": score["exact"],
                            "fallback_chunks": sum(item[2] for item in predicted_chunks)})

    count = max(len(inputs), 1)
    report = {
        "split": args.split,
        "unique_inputs": len(inputs),
        "decoded_chunks": totals["chunks"],
        "sentences_with_fallback": totals["fallback_sentences"],
        "chunks_with_fallback": totals["fallback_chunks"],
        "macro_word_precision": macro["precision"] / count,
        "macro_word_recall": macro["recall"] / count,
        "macro_word_f1": macro["f1"] / count,
        "perfect_sentence_match_any_reference": exact / count,
        "method": "Split input at existing whitespace, predict each chunk independently, then join predictions with one space.",
        "limitations": ["Dev-only candidate; not evaluated on test.",
                        "Existing whitespace is assumed to mark stable chunk boundaries.",
                        "Chunk predictions have less sentence context than the full-sentence model."],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "whitespace_chunk_candidate_dev.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUTPUT_DIR / "whitespace_chunk_candidate_dev.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
