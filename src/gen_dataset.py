#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

from datasets import load_dataset

HEADER = [
    "model",
    "question",
    "prediction",
    "ground_truth",
    "verdict",
    "reason",
    "tags",
    "_raw_judge",
]

LETTERS = ["A", "B", "C", "D"]


def _get_options(example):
    """
    Try to recover the four answer options from a MedQA example.

    Different MedQA variants use slightly different field names, so we
    probe several common patterns.
    """
    # 1) Direct list field
    options = example.get("options")
    if isinstance(options, (list, tuple)) and len(options) >= 4:
        return list(options)[:4]
    # 1b) Dict mapping letters -> option text (MedQA-USMLE-4-options format)
    if isinstance(options, dict):
        return [options.get(letter, "") for letter in LETTERS]

    # 2) Common per-option field name patterns
    candidate_key_sets = [
        ["option_A", "option_B", "option_C", "option_D"],
        ["opa", "opb", "opc", "opd"],
        ["A", "B", "C", "D"],
        ["choice_A", "choice_B", "choice_C", "choice_D"],
    ]

    for keys in candidate_key_sets:
        vals = [example.get(k, "") for k in keys]
        # If at least one option has non-empty text, assume this pattern is correct
        if any(v not in ("", None) for v in vals):
            return vals

    # Fallback: empty strings (so at least the script doesn't crash)
    return ["", "", "", ""]


def build_question_text(example) -> str:
    """Return only the question stem (no options)."""
    stem = example.get("question") or example.get("problem") or ""
    return stem


def get_correct_letter(example):
    """
    Extract the correct answer letter (A–D) from a MedQA example.

    MedQA variants may store the answer as:
    - an index (0–3) in `answer_idx`, or
    - a letter ('A'–'D') in `answer_idx` / `answer` / `answer_choice`.
    """
    answer_idx = example.get("answer_idx")
    answer_letter = example.get("answer") or example.get("answer_choice")

    # Case 1: answer_idx is a string (either a letter or a digit)
    if isinstance(answer_idx, str):
        cleaned = answer_idx.strip().upper()
        if cleaned in LETTERS:
            return cleaned
        if cleaned.isdigit():
            idx = int(cleaned)
            if 0 <= idx < len(LETTERS):
                return LETTERS[idx]

    # Case 2: answer_idx is an integer index
    if isinstance(answer_idx, int):
        if 0 <= answer_idx < len(LETTERS):
            return LETTERS[answer_idx]

    # Case 3: fall back to explicit letter field
    if isinstance(answer_letter, str):
        cleaned = answer_letter.strip().upper()
        if cleaned in LETTERS:
            return cleaned

    # If we can't determine the correct option, signal to skip
    return None


def get_correct_answer_text(example, options):
    """
    Get the correct answer *text* for a MedQA example.

    Prefer the explicit `answer` field; if missing, fall back to
    using the correct letter to index into the options.
    """
    answer = example.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    correct_letter = get_correct_letter(example)
    if correct_letter is None:
        return None

    try:
        idx = LETTERS.index(correct_letter)
    except ValueError:
        return None

    if 0 <= idx < len(options):
        return str(options[idx]).strip()

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/medqa_mcq_synthetic_safety.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1000,
        help="Number of MedQA questions to use (each creates 4 rows).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="synthetic-medqa-mcq",
        help="Value to put in the `model` column.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="HuggingFace datasets cache directory.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="train",
        cache_dir=str(args.cache_dir),
    )

    if args.num_questions > 0 and args.num_questions < len(ds):
        ds = ds.select(range(args.num_questions))

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for ex in ds:
            question_text = build_question_text(ex)
            options = _get_options(ex)
            correct_text = get_correct_answer_text(ex, options)
            if correct_text is None:
                # Skip examples where we can't determine the correct answer text
                continue

            for opt_text in options:
                opt_clean = str(opt_text).strip()
                verdict = "SAFE" if opt_clean == correct_text else "UNSAFE"
                row = [
                    args.model_name,  # model
                    question_text,    # question
                    opt_clean,        # prediction (answer text)
                    correct_text,     # ground_truth (correct answer text)
                    verdict,          # verdict
                    "",               # reason (empty)
                    "",               # tags (empty)
                    "",               # _raw_judge (empty)
                ]
                writer.writerow(row)


if __name__ == "__main__":
    main()
