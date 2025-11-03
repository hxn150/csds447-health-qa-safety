import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import Dataset


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return tokens


def token_overlap_metrics(pred: str, gold: str) -> tuple[float, float, float]:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not gold_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    overlap = sum(common.values())

    precision = overlap / sum(pred_counter.values()) if pred_counter else 0.0
    recall = overlap / sum(gold_counter.values()) if gold_counter else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def load_medqa_lookup() -> dict[str, dict]:
    cache_root = Path("data/cache")
    arrow_files = sorted(
        cache_root.glob("GBaker___med_qa-usmle-4-options/**/med_qa-usmle-4-options-test.arrow")
    )
    if not arrow_files:
        return {}

    dataset = Dataset.from_file(str(arrow_files[0]))
    lookup = {}
    for row in dataset:
        lookup[normalize_text(row["question"])] = {
            "options": row["options"],
            "answer_idx": row["answer_idx"],
        }
    return lookup


MC_LOOKUP_LOADERS: dict[str, Callable[[], dict]] = {
    "medqa": load_medqa_lookup,
}


def determine_choice(prediction: str, options: dict) -> tuple[str | None, float]:
    best_choice = None
    best_f1 = -1.0
    for letter, option_text in options.items():
        _, _, f1 = token_overlap_metrics(prediction, option_text)
        if f1 > best_f1:
            best_f1 = f1
            best_choice = letter
    return best_choice, best_f1


def evaluate_file(path: Path, dataset_name: str, mc_context: dict[str, dict]) -> dict[str, float]:
    frame = pd.read_csv(path)
    
    def map_pubmedqa_label(text: str) -> str:
        import re
        if not isinstance(text, str):
            return ""
        for m in re.finditer(r"\b(yes|no|maybe)\b", text.lower()):
            return m.group(1)
        lower = text.lower()
        if "yes" in lower: return "yes"
        if "no" in lower: return "no"
        if "maybe" in lower: return "maybe"
        return ""
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_hits = []

    mc_lookup = mc_context.get(dataset_name, {})

    for _, row in frame.iterrows():
        prediction = row.get("prediction", "")
        # Robustness for PubMedQA free-form answers
        if dataset_name == "pubmedqa":
            mapped = map_pubmedqa_label(str(prediction))
            if mapped:
                prediction = mapped
        gold = row.get("ground_truth", "")
        question = row.get("question", "")

        is_exact = normalize_text(prediction) == normalize_text(gold)

        if mc_lookup:
            mc_row = mc_lookup.get(normalize_text(question))
            if mc_row:
                predicted_choice, _ = determine_choice(prediction, mc_row["options"])
                accuracy_hits.append(1.0 if predicted_choice == mc_row["answer_idx"] else 0.0)
            else:
                accuracy_hits.append(1.0 if is_exact else 0.0)
        else:
            accuracy_hits.append(1.0 if is_exact else 0.0)

        precision, recall, f1 = token_overlap_metrics(prediction, gold)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    accuracy = sum(accuracy_hits) / len(accuracy_hits) if accuracy_hits else 0.0
    precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    rmse = math.sqrt(sum((1.0 - score) ** 2 for score in f1_scores) / len(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rmse": rmse,
        "count": len(frame),
    }


def collect_prediction_files(output_dir: Path) -> dict[str, list[Path]]:
    matrix: dict[str, list[Path]] = {}
    for csv_path in output_dir.glob("pred_*.csv"):
        # Skip combined rollups; evaluate per-model files only
        if csv_path.stem.endswith("_combined"):
            continue
        name_parts = csv_path.stem.split("_", 2)
        if len(name_parts) < 3:
            continue
        dataset = name_parts[1]
        matrix.setdefault(dataset, []).append(csv_path)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs", help="Directory with prediction CSVs")
    parser.add_argument("--dest", default="outputs/metrics.csv", help="Where to write the metrics summary")
    args = parser.parse_args()

    output_dir = Path(args.outputs)
    output_dir.mkdir(exist_ok=True)

    datasets_to_files = collect_prediction_files(output_dir)
    mc_context = {}
    for dataset in datasets_to_files:
        loader = MC_LOOKUP_LOADERS.get(dataset)
        if loader:
            mc_context[dataset] = loader()

    rows = []

    for dataset, paths in sorted(datasets_to_files.items()):
        for csv_path in paths:
            model_name = csv_path.stem.split("_", 2)[-1].replace("_", "/")
            metrics = evaluate_file(csv_path, dataset, mc_context)
            row = {
                "dataset": dataset,
                "model": model_name,
                **metrics,
            }
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.sort_values(["dataset", "accuracy"], ascending=[True, False], inplace=True)
    summary.to_csv(args.dest, index=False)
    print(f"[✓] Wrote metrics → {args.dest}")


if __name__ == "__main__":
    main()
