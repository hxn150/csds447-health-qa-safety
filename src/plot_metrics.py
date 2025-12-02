import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def load_data():
    metrics_path = OUTPUTS / "metrics.csv"
    judge_path = OUTPUTS / "metrics_judge.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not judge_path.exists():
        raise FileNotFoundError(f"Missing judge metrics file: {judge_path}")

    metrics = pd.read_csv(metrics_path)
    judge = pd.read_csv(judge_path)
    return metrics, judge

def plot_safety(judge: pd.DataFrame):
    rate_cols = ["safe_rate", "partially_safe_rate", "unsafe_rate", "unsure_rate"]
    colors = {
        "safe_rate": "green",
        "partially_safe_rate": "yellow",
        "unsafe_rate": "red",
        "unsure_rate": "blue",
    }
    datasets = sorted(judge["dataset"].unique())

    for dataset in datasets:
        df_ds = judge[judge["dataset"] == dataset].copy()

        # Order so each model stands next to its baseline
        def base_name(model: str) -> str:
            suffix = "/baseline"
            return model[:-len(suffix)] if model.endswith(suffix) else model

        df_ds["base_name"] = df_ds["model"].apply(base_name)
        df_ds["is_baseline"] = df_ds["model"].str.endswith("/baseline").astype(int)
        df_ds.sort_values(
            by=["base_name", "is_baseline"],
            ascending=[True, False],  # baseline first, then fine-tuned
            inplace=True,
        )

        models = df_ds["model"].tolist()
        x = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(10, 6))

        bottom = np.zeros(len(models))
        for col in rate_cols:
            values = df_ds[col].values
            ax.bar(x, values, bottom=bottom, label=col, color=colors.get(col, None))
            bottom += values

        ax.set_title(f"Safety verdict distribution — {dataset}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        df_ds.drop(columns=["base_name", "is_baseline"], inplace=True)
        out_path = OUTPUTS / f"safety_{dataset}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

def main():
    OUTPUTS.mkdir(exist_ok=True)
    metrics, judge = load_data()
    plot_safety(judge)


if __name__ == "__main__":
    main()
