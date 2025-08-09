#!/usr/bin/env python3
"""Plot AUROC comparison of baseline methods for CoD vs CoT.

For each dataset we show grouped bars (CoD and CoT) for the three baseline
methods: probas-min-bl, probas-mean-bl, tokensar-bl.

Figure is saved to Results_Analysis/results/comparison/auroc_cod_vs_cot.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# -----------------------------------------------------------------------------
# Helpers ----------------------------------------------------------------------
# -----------------------------------------------------------------------------


def load_aggregated_json(path: Path) -> dict:
    """Return dict from aggregated JSON file at *path* or empty dict if missing."""
    if not path.exists():
        print(f"Warning: missing {path}")
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"Warning: could not decode {path}")
        return {}


# -----------------------------------------------------------------------------
# Main -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="AUROC comparison CoD vs CoT (baselines)")
    parser.add_argument("--cod-dir", type=str, default="results/cod/auroc/aggregated",
                        help="Directory with aggregated AUROC JSONs for CoD")
    parser.add_argument("--cot-dir", type=str, default="results/cot/auroc/aggregated",
                        help="Directory with aggregated AUROC JSONs for CoT")
    parser.add_argument("--out", type=str, default="results/comparison/auroc_cod_vs_cot.png",
                        help="Path to save the generated figure (relative to project root)")
    parser.add_argument("--baselines", nargs="*", default=[
                        "probas-min-bl", "probas-mean-bl", "token-sar-bl"],
                        help="List of baseline metric names to compare")
    args = parser.parse_args()

    cod_dir = project_root / args.cod_dir
    cot_dir = project_root / args.cot_dir
    baselines = args.baselines

    # Derive dataset list from CoD directory (assuming same datasets for CoT)
    cod_jsons = sorted(cod_dir.glob("*_auroc.json"))
    if not cod_jsons:
        print(f"No aggregated JSONs found in {cod_dir}")
        sys.exit(1)
    datasets = [p.name.replace("_auroc.json", "") for p in cod_jsons]

    rows: list[dict] = []
    for paradigm, root in [("CoD", cod_dir), ("CoT", cot_dir)]:
        for ds in datasets:
            data = load_aggregated_json(root / f"{ds}_auroc.json")
            for metric in baselines:
                stats = data.get(metric, {})
                scores = stats.get("individual_scores", [])
                if not scores:
                    # fall back to mean if individual not present
                    mean = stats.get("mean_auroc", None)
                    if mean is not None:
                        scores = [mean]
                for s in scores:
                    rows.append({
                        "dataset": ds,
                        "baseline": metric,
                        "paradigm": paradigm,
                        "auroc": s,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data to plot. Check directories and baseline names.")
        sys.exit(1)

    # Clean labels (strip prefixes/suffixes like in leaderboard script)
    label_map = {
        "probas-min-bl": "probas-min",
        "probas-mean-bl": "probas-mean",
        "token-sar-bl": "token-sar",
    }
    df["baseline_label"] = df["baseline"].map(label_map).fillna(df["baseline"])

    sns.set(style="whitegrid")
    # Determine subplot grid (max 6 per row like leaderboard)
    n_ds = len(datasets)
    n_cols = min(3, n_ds)
    n_rows = (n_ds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    axes = axes.flatten()

    palette = {"CoD": "#1f77b4", "CoT": "#ff7f0e"}

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        if sub.empty:
            ax.axis("off")
            continue
        # compute mean AUROC per baseline/paradigm for bar heights
        grp = sub.groupby(["baseline_label", "paradigm"], as_index=False)["auroc"].mean()
        sns.barplot(data=grp, x="baseline_label", y="auroc", hue="paradigm",
                    palette=palette, ax=ax, edgecolor="black", width=0.6)
        ax.set_title(ds)
        ax.set_xlabel("")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, linestyle="--", color="grey")
        # annotate bars with values
        for container in ax.containers:
            ax.bar_label(container, fmt="{:.2f}", padding=3, fontsize=8)
    # Remove any unused axes
    for ax in axes[len(datasets):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", title="Aggregationsmetriken Baseline")
    plt.tight_layout()

    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved AUROC CoD vs CoT comparison to {out_path}")


if __name__ == "__main__":
    main()
