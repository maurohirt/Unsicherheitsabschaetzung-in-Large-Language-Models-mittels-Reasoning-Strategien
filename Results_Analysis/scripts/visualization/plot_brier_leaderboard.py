#!/usr/bin/env python3
"""
Plot a Brier Score "leaderboard" across datasets: violin distributions of run-wise Brier scores,
mean diamonds with CIs, and jittered points showing raw variability.
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# ensure project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import load_yaml_with_imports

def main():
    project_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml_with_imports(project_root / 'configs' / 'brier_config.yaml')
    datasets = cfg.get('datasets', [])
    raw_metrics = cfg.get('metrics', [])
    # resolve '@' group refs
    metrics = []
    for item in raw_metrics:
        if isinstance(item, str) and item.startswith('@'):
            grp = item[1:]
            for e in cfg.get(grp, []):
                if isinstance(e, dict) and 'name' in e:
                    metrics.append(e['name'])
                elif isinstance(e, str):
                    metrics.append(e)
        else:
            metrics.append(item)
    # dedupe
    seen = set()
    metrics = [m for m in metrics if not (m in seen or seen.add(m))]
    # groups for legend
    group_keys = ['baseline_methods', 'cot_methods', 'true_probability_methods', 'self_probing_methods']
    groups = {g: [e['name'] for e in cfg.get(g, [])] for g in group_keys}
    fam_map = {m: g for g, ms in groups.items() for m in ms}
    families = list(groups.keys())
    runs = cfg.get('runs', [])
    # collect data
    rows = []
    results_dir = project_root / cfg.get('results_path', 'results/cot/brier')
    for ds in datasets:
        ds_dir = results_dir / ds
        agg_path = ds_dir / 'aggregated' / f"{ds}_brier.json"
        if not agg_path.exists():
            print(f"Skipping {ds}: missing aggregated {agg_path}")
            continue
        agg = json.loads(agg_path.read_text())
        for run in runs:
            run_file = ds_dir / f"run_{run}" / f"{ds}_brier.json"
            if not run_file.exists():
                continue
            run_data = json.loads(run_file.read_text())
            for m in metrics:
                if m not in run_data:
                    continue
                score = run_data[m].get('score', np.nan)
                stats = agg.get(m, {})
                mean = stats.get('mean_brier', np.nan)
                ci = stats.get('confidence_interval', [np.nan, np.nan])
                ci_low, ci_high = ci if len(ci)==2 else (np.nan, np.nan)
                fam = fam_map.get(m, 'other')
                rows.append({'dataset': ds, 'metric': m, 'score': score,
                             'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high,
                             'family': fam})
    if not rows:
        print("No data to plot.")
        return
    df = pd.DataFrame(rows)
    # simplify x-labels
    metric_labels = []
    for m in metrics:
        fam = fam_map.get(m, '')
        if fam in ['baseline_methods', 'cot_methods']:
            label = m[len('probas-'):] if m.startswith('probas-') else m
            if label.endswith('-bl'):
                label = label[:-3]
        elif fam == 'true_probability_methods':
            label = m[len('p-true-'):] if m.startswith('p-true-') else m
            if label == 'bl': label = 'baseline'
        elif fam == 'self_probing_methods':
            label = m[len('self-probing-'):] if m.startswith('self-probing-') else m
            if label == 'bl': label = 'baseline'
        else:
            label = m
        metric_labels.append(label)
    # color palette per family
    pal = sns.color_palette('tab10', n_colors=len(families))
    fam_pal = dict(zip(families, pal))
    col_list = [fam_pal.get(fam_map.get(m), (0.5,0.5,0.5)) for m in metrics]
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(12, 12), sharey=False)
    axes = axes.flatten()
    # plot panels
    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset']==ds]
        scores = [sub[sub['metric']==m]['score'].tolist() for m in metrics]
        parts = ax.violinplot(scores, positions=list(range(len(metrics))), widths=0.8, showextrema=False)
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(col_list[idx]); pc.set_edgecolor('black'); pc.set_alpha(0.8)
        # raw points via manual jitter to avoid categorical axis issues
        for idx, m in enumerate(metrics):
            ys = sub[sub['metric']==m]['score'].tolist()
            if not ys:
                continue
            xs = np.random.normal(loc=idx, scale=0.08, size=len(ys))
            ax.scatter(xs, ys, color='black', s=6, alpha=0.6)

        for i, m in enumerate(metrics):
            msub = sub[sub['metric']==m]
            if msub.empty: continue
            mean = msub['mean'].iloc[0]; low = msub['ci_low'].iloc[0]; high = msub['ci_high'].iloc[0]
            ax.scatter(i, mean, color='black', marker='D', zorder=10)
            ax.vlines(i, low, high, color='black', linewidth=1)
        ax.axhline(0, linestyle='--', color='grey')
        ax.set_title(ds)
        ax.set_xlabel('')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.set_ylabel('Brier Score')
    # legend
    legend_ax = axes[len(datasets)]; legend_ax.axis('off')
    label_map = {
        'baseline_methods': 'baseline',
        'cot_methods': 'CoT',
        'true_probability_methods': 'p(true)',
        'self_probing_methods': 'self-probing'
    }
    handles = [Patch(facecolor=fam_pal[f], label=label_map.get(f, f)) for f in families]
    legend_ax.legend(handles=handles, loc='center')
    plt.tight_layout()
    out = project_root / Path(cfg.get('results_path', 'results/cot/brier')).parent / 'figures' / 'brier_leaderboard.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"Saved Brier Score leaderboard to {out}")

if __name__ == '__main__':
    main()
