#!/usr/bin/env python3
"""
Plot Reliability Diagram for P(True) family.
One plot per dataset with 5 curves: baseline (p-true-bl) and four variants.
Saves under results/cot/figures/reliability/{dataset}/ptrue_strategies.png
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
    # load datasets
    ds_cfg = load_yaml_with_imports(project_root / 'configs' / 'datasets.yaml')
    datasets = ds_cfg.get('datasets', [])

    # define metrics and styles
    metrics = ['p-true-bl', 'p-true-allsteps', 'p-true-keystep', 'p-true-allkeywords', 'p-true-keykeywords']
    styles = ['-', '--', '-.', ':', (0, (5, 1))]  # unique line styles
    palette = sns.color_palette('tab10')
    base_color = palette[2]  # choose green for p-true

    # uniform solid lines for all strategies
    style_map = {m: '-' for m in metrics}
    # color: baseline black, variants colored
    variant_colors = sns.color_palette('tab10', n_colors=len(metrics)-1)
    color_map = {metrics[0]: 'black'}
    for idx, m in enumerate(metrics[1:]):
        color_map[m] = variant_colors[idx]
    label_map = {m: m.replace('p-true-', '') for m in metrics}

    for ds in datasets:
        ece_path = project_root / 'results' / 'cot' / 'ece' / ds / 'aggregated' / f"{ds}_ece.json"
        if not ece_path.exists():
            print(f"Skipping {ds}: missing {ece_path}")
            continue
        agg = json.loads(ece_path.read_text())

        plt.figure(figsize=(6, 6))
        sns.set(style='whitegrid')
        # diagonal
        plt.plot([0, 1], [0, 1], '--', color='gray')
        legend_handles = []
        for m in metrics:
            if m not in agg:
                continue
            conf = agg[m]['mean_bin_confidence']
            acc = agg[m]['mean_bin_accuracy']
            lw = 2.5 if m.endswith('-bl') else 1.5
            plt.plot(conf, acc, linestyle='-', color=color_map[m], linewidth=lw)
            # legend entry
            ece = agg[m].get('mean_ece', agg[m].get('score'))
            ci = agg[m].get('confidence_interval', [ece, ece])
            half_ci = (ci[1] - ci[0]) / 2
            handle = Line2D([0], [0], color=color_map[m], linestyle='-', linewidth=lw,
                            label=f"{label_map[m]} (ECE = {ece:.3f} ± {half_ci:.3f})")
            legend_handles.append(handle)

        plt.xlabel('Average Confidence')
        plt.ylabel('Empirical Accuracy')
        plt.title(f'{ds} – P(True) Calibration Curves')
        plt.legend(handles=legend_handles, loc='best', fontsize=9)
        plt.tight_layout()

        out_dir = project_root / 'results' / 'cot' / 'figures' / 'reliability' / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'ptrue_strategies.png'
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved {out_file}")


if __name__ == '__main__':
    main()
