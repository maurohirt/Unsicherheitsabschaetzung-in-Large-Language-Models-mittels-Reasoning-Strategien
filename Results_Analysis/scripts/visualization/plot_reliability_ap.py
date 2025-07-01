#!/usr/bin/env python3
"""
Plot Reliability Diagram for Aggregated Probabilities (AP family).
One plot per dataset with 6 curves: mean/bl, min/bl, token-sar/bl.
Saves under results/cot/figures/reliability/{dataset}/aggregated_probabilities.png
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

# ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import load_yaml_with_imports

def lighten(color, factor=0.5):
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * factor)


def main():
    project_root = Path(__file__).resolve().parents[2]
    # load dataset names
    ds_cfg = load_yaml_with_imports(project_root / 'configs' / 'datasets.yaml')
    datasets = ds_cfg.get('datasets', [])
    # define stats and metrics
    stats = ['probas-mean', 'probas-min', 'token-sar']
    metrics = [f"{s}-bl" for s in stats] + stats
    # color palette for stats
    pal = sns.color_palette('tab10', n_colors=len(stats))
    color_map = {s: pal[i] for i, s in enumerate(stats)}
    # labels
    label_map = {'probas-mean': 'mean', 'probas-min': 'min', 'token-sar': 'token-sar'}

    for ds in datasets:
        # load aggregated ECE results
        ece_path = project_root / 'results' / 'cot' / 'ece' / ds / 'aggregated' / f"{ds}_ece.json"
        if not ece_path.exists():
            print(f"Skipping {ds}: missing {ece_path}")
            continue
        agg = json.loads(ece_path.read_text())

        plt.figure(figsize=(6, 6))
        sns.set(style='whitegrid')
        # perfect calibration line
        plt.plot([0,1], [0,1], '--', color='gray', label=None)

        legend_handles = []
        # plot each stat baseline and CoT
        for s in stats:
            for kind, ls in [('baseline','-'), ('CoT','--')]:
                key = f"{s}-bl" if kind=='baseline' else s
                if key not in agg:
                    continue
                conf = agg[key]['mean_bin_confidence']
                acc = agg[key]['mean_bin_accuracy']
                color = lighten(color_map[s], factor=0.5) if kind=='baseline' else color_map[s]
                plt.plot(conf, acc, linestyle=ls, color=color, linewidth=2)
                # legend entry
                ece = agg[key].get('mean_ece', agg[key].get('score'))
                ci = agg[key].get('confidence_interval', [np.nan, np.nan])
                half_ci = (ci[1] - ci[0]) / 2 if len(ci)==2 else np.nan
                lbl = f"{label_map[s]}-{kind} (ECE = {ece:.3f} ± {half_ci:.3f})"
                handle = Line2D([0], [0], marker='s', color=color,
                                markerfacecolor=color, linestyle='None', markersize=8, label=lbl)
                legend_handles.append(handle)

        plt.xlabel('Average Confidence')
        plt.ylabel('Empirical Accuracy')
        plt.title(f'{ds} – Aggregated Probabilities Calibration Curves')
        plt.legend(handles=legend_handles, loc='best', fontsize=9)
        plt.tight_layout()

        out_dir = project_root / 'results' / 'cot' / 'figures' / 'reliability' / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'aggregated_probabilities.png'
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved {out_file}")

if __name__ == '__main__':
    main()
