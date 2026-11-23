"""Segment one or more Roman-transliterated Sanskrit lines with the saved model."""

import argparse
import sys
import unicodedata
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "module_03_word_segmentation" / "best_model.pt"
sys.path.insert(0, str(SCRIPT_DIR))
from train_character_transformer import (  # noqa: E402
    BOS, EOS, PAD, CharacterTransformer, greedy_decode,
)


def segment(line, model, tokens, token_to_id, device, max_length):
    line = unicodedata.normalize("NFC", line)
    if len(line) > max_length:
        raise ValueError(f"Input has {len(line)} characters; maximum is {max_length}.")
    unknown = sorted(set(line) - set(token_to_id))
    if unknown:
        raise ValueError(f"Input contains characters outside the training vocabulary: {unknown}")
    source_ids = [token_to_id[ch] for ch in line] + [token_to_id[EOS]]
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    return greedy_decode(model, source, token_to_id, tokens, max_length)[0]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="One Roman-transliterated Sanskrit sentence.")
    group.add_argument("--input-file", type=Path, help="UTF-8 text file, one sentence per line.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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
        print(segment(line, model, tokens, token_to_id, device, max_length))


if __name__ == "__main__":
    main()
