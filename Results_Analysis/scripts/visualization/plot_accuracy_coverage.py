#!/usr/bin/env python3
"""
Plot Accuracy vs Coverage for Math & Logic datasets (asdiv & hotpotqa).
Metrics: probas-mean-bl (baseline), probas-min, token-sar.
"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# ensure project root on sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.utils.config_loader import load_yaml_with_imports


def load_confidences(path: Path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for l in lines:
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    project_root = Path(__file__).resolve().parents[2]
    # load config
    acc_cfg = load_yaml_with_imports(project_root / 'configs' / 'accuracy_config.yaml')
    data_dir = project_root / acc_cfg['data_path']
    model = acc_cfg['model']
    runs = acc_cfg['runs']

    datasets = ['ASDiv', 'hotpotQA']
    metrics = ['probas-min-bl', 'probas-min', 'probas-mean-bl', 'probas-mean', 'token-sar-bl', 'token-sar']
    label_map = {
        'probas-mean-bl': 'Mean BL',
        'probas-mean': 'Mean',
        'probas-min-bl': 'Min BL',
        'probas-min': 'Min',
        'token-sar-bl': 'Token-SAR BL',
        'token-sar': 'Token-SAR'
    }
    # define color palette per base metric (light baseline, dark variant)
    base_names = ['probas-mean', 'probas-min', 'token-sar']
    palette = sns.color_palette('tab10', n_colors=len(base_names))
    base_color_map = {bn: palette[i] for i, bn in enumerate(base_names)}

    for ds in datasets:
        # build label map per run
        label_map_by_run = {}
        for r in runs:
            lbl_rows = load_confidences(data_dir / f"run_{r}" / model / ds / 'output_v1_w_labels.json')
            label_map_by_run[r] = {str(row.get("question", "")).strip(): row.get("label", False) for row in lbl_rows}
        curves = {}
        for m in metrics:
            curves_by_run = []
            for r in runs:
                p = data_dir / f"run_{r}" / model / ds / 'confidences' / f"output_v1_{m}.json"
                rows = load_confidences(p)
                lbl_map = label_map_by_run.get(r, {})
                confs, corrs = [], []
                for row in rows:
                    confs.append(row.get('confidence', np.nan))
                    q_raw = row.get('question', '')
                    q = str(q_raw).strip()
                    ans = str(row.get('llm answer', '')).strip()
                    gt = str(row.get('correct answer', '')).strip()
                    if q in lbl_map:
                        corr = lbl_map[q]
                    else:
                        corr = (ans == gt)
                    corrs.append(int(corr))
                if not confs:
                    continue
                arr_conf = np.array(confs)
                arr_corr = np.array(corrs, dtype=int)
                order = np.argsort(-arr_conf)
                sorted_corr = arr_corr[order]
                n = len(sorted_corr)
                cum = np.cumsum(sorted_corr)
                cover = np.concatenate(([0], np.arange(1, n+1) / n))
                acc = np.concatenate(([1], cum / np.arange(1, n+1)))
                curves_by_run.append((cover, acc))
            if not curves_by_run:
                continue
            # average runs via interpolation on a common coverage grid
            common_cov = np.linspace(0, 1, 100)
            acc_interp = [np.interp(common_cov, cov, ac) for cov, ac in curves_by_run]
            mean_acc = np.mean(acc_interp, axis=0)
            curves[m] = (common_cov, mean_acc)


        # plot
        plt.figure(figsize=(6, 6))
        sns.set(style='whitegrid')
        for m in metrics:
            if m not in curves:
                continue
            cov, ac = curves[m]
            plt.plot(cov, ac,
                 color=base_color_map.get(m.replace('-bl',''), 'grey'),
                 linewidth=2,
                 alpha=0.5 if m.endswith('-bl') else 1.0,
                 label=label_map.get(m, m))
        plt.xlabel('Coverage')
        plt.ylabel('Accuracy')
        plt.title(f'{ds} – Accuracy vs Coverage')
        plt.legend(title='Method', loc='upper right')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.tight_layout()

        out_dir = project_root / 'results' / 'cot' / 'figures' / 'accuracy_coverage' / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'accuracy_coverage_{ds}.png'
        plt.savefig(out_file, dpi=300)
        print(f"Saved {out_file}")
        plt.close()

if __name__ == '__main__':
    main()
