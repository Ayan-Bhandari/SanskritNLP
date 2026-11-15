"""Show how the Module 1 deterministic normalizer handles common cases."""

from pathlib import Path
import sys


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from normalization import normalize_text


EXAMPLES = [
    ("NFC composition (general Unicode)", "e\u0301", "é"),
    ("Repeated spaces", "रामः   वनम्", "रामः वनम्"),
    ("Tab between words", "रामः\tवनम्", "रामः वनम्"),
    ("Windows CRLF", "रामः\r\nवनम्", "रामः\nवनम्"),
    ("Preserved line break", "रामः\nवनम्", "रामः\nवनम्"),
    ("Preserved blank line", "रामः\n\nवनम्", "रामः\n\nवनम्"),
    ("Leading and trailing spaces", "  रामः  ", "रामः"),
    ("Punctuation preserved", "रामः॥", "रामः॥"),
    ("Missing value", None, None),
]


def main():
    for label, input_text, expected_text in EXAMPLES:
        actual_text = normalize_text(input_text)
        if actual_text == expected_text:
            result = "matches"
        else:
            result = "DIFFERS"

        print(label)
        print(f"  input:    {input_text!r}")
        print(f"  output:   {actual_text!r}")
        print(f"  expected: {expected_text!r} ({result})")


if __name__ == "__main__":
    main()
