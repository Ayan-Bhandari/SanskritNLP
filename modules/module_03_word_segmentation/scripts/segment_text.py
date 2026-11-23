"""Segment one or more Roman-transliterated Sanskrit lines with the saved model."""

import argparse
import sys
import unicodedata
from pathlib import Path

import torch
from rapidfuzz.distance import Levenshtein


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "module_03_word_segmentation" / "best_model.pt"
sys.path.insert(0, str(SCRIPT_DIR))
from train_character_transformer import (  # noqa: E402
    EOS, MAX_CHARACTER_EDIT_RATIO, PAD, ROUTE_TO_CHUNKS_ABOVE_LENGTH,
    CharacterTransformer, greedy_decode,
)


def segment(line, model, tokens, token_to_id, device, max_length, use_fallback=True):
    line = unicodedata.normalize("NFC", line)
    if len(line) > max_length:
        raise ValueError(f"Input has {len(line)} characters; maximum is {max_length}.")
    unknown = sorted(set(line) - set(token_to_id))
    if unknown:
        raise ValueError(f"Input contains characters outside the training vocabulary: {ascii(unknown)}")
    if len(line) > ROUTE_TO_CHUNKS_ABOVE_LENGTH:
        chunks = line.split()
        sequences = [[token_to_id[ch] for ch in chunk] + [token_to_id[EOS]] for chunk in chunks]
        source = torch.full((len(sequences), max(map(len, sequences))), token_to_id[PAD], dtype=torch.long)
        for index, sequence in enumerate(sequences):
            source[index, :len(sequence)] = torch.tensor(sequence)
        decoded = greedy_decode(model, source.to(device), token_to_id, tokens, max_length)
        safe_chunks = []
        fallback = False
        for chunk, prediction in zip(chunks, decoded):
            predicted_chars = "".join(prediction.split())
            edit_ratio = Levenshtein.distance(chunk, predicted_chars) / max(len(chunk), 1)
            use_identity = use_fallback and edit_ratio > MAX_CHARACTER_EDIT_RATIO
            safe_chunks.append(chunk if use_identity else prediction)
            fallback = fallback or use_identity
        return " ".join(safe_chunks), fallback

    source_ids = [token_to_id[ch] for ch in line] + [token_to_id[EOS]]
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    prediction = greedy_decode(model, source, token_to_id, tokens, max_length)[0]
    source_chars = "".join(line.split())
    predicted_chars = "".join(prediction.split())
    edit_ratio = Levenshtein.distance(source_chars, predicted_chars) / max(len(source_chars), 1)
    fallback = use_fallback and edit_ratio > MAX_CHARACTER_EDIT_RATIO
    return (line if fallback else prediction), fallback


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="One Roman-transliterated Sanskrit sentence.")
    group.add_argument("--input-file", type=Path, help="UTF-8 text file, one sentence per line.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Return the raw model output even when its character edits exceed the safety threshold.")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    tokens = checkpoint["tokens"]
    token_to_id = {token: index for index, token in enumerate(tokens)}
    max_length = checkpoint["config"]["max_length"]
    model = CharacterTransformer(len(tokens), max_length).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.text is not None:
        lines = [args.text]
    else:
        lines = args.input_file.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            print("")
            continue
        prediction, fallback = segment(line, model, tokens, token_to_id, device, max_length,
                                       use_fallback=not args.no_fallback)
        if fallback:
            print("Warning: generated text was too divergent; returned the input unchanged.", file=sys.stderr)
        print(prediction)


if __name__ == "__main__":
    main()
