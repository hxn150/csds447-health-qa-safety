#!/usr/bin/env python
"""
Generate a safety QA dataset with two columns: question,label.

- SAFE questions come from PubMedQA (qiaojin/PubMedQA, pqa_labeled).
- UNSAFE questions come from a local harmful prompts file (e.g. src/harmful.txt).
"""

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset


def iter_pubmedqa_questions(
    split: str,
    limit: Optional[int],
    cache_dir: Path,
) -> Iterable[str]:
    """Yield cleaned PubMedQA question strings."""
    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split=split,
        cache_dir=str(cache_dir),
    )
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        question = row.get("question", "")
        question = str(question).strip()
        if not question:
            continue
        yield question


def iter_harmful_questions(path: Path) -> Iterable[str]:
    """Yield harmful questions/prompts from a text file, one per non-empty line."""
    with path.open("r", encoding="utf-8") as f:
        i = 0
        for line in f:
            text = line.strip()
            if not text:
                continue
            # Remove known special tokens if present
            text = text.replace("<|eot_id|>", "").strip()
            # Strip simple surrounding quotes
            if (text.startswith('"') and text.endswith('"')) or (
                text.startswith("'") and text.endswith("'")
            ):
                text = text[1:-1].strip()
            if text:
                yield text
            i+=1
            if i > 1000:
                break

def build_dataset(
    output_path: Path,
    pubmedqa_split: str = "train",
    max_safe: Optional[int] = None,
    cache_dir: Path = Path("data/cache"),
    harmful_path: Path = Path("src/harmful.txt"),
) -> None:
    """Write a CSV with columns question,label using SAFE and UNSAFE sources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "label"])

        # SAFE examples from PubMedQA
        for question in iter_pubmedqa_questions(
            split=pubmedqa_split,
            limit=max_safe,
            cache_dir=cache_dir,
        ):
            writer.writerow([question, "SAFE"])

        # UNSAFE examples from harmful.txt
        for question in iter_harmful_questions(harmful_path):
            writer.writerow([question, "UNSAFE"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a simple safety dataset with two columns: question,label. "
            "PubMedQA questions are labeled SAFE and questions from src/harmful.txt "
            "are labeled UNSAFE."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/safety_qa.csv"),
        help="Output CSV path (default: data/datasets/safety_qa.csv).",
    )
    parser.add_argument(
        "--pubmedqa-split",
        type=str,
        default="train",
        help="PubMedQA split to use (train/validation/test).",
    )
    parser.add_argument(
        "--max-safe",
        type=int,
        default=None,
        help="Optional maximum number of SAFE (PubMedQA) questions to include.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="HuggingFace datasets cache directory.",
    )
    parser.add_argument(
        "--harmful-path",
        type=Path,
        default=Path("src/harmful.txt"),
        help="Path to harmful questions file (one per line).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        output_path=args.output,
        pubmedqa_split=args.pubmedqa_split,
        max_safe=args.max_safe,
        cache_dir=args.cache_dir,
        harmful_path=args.harmful_path,
    )


if __name__ == "__main__":
    main()

