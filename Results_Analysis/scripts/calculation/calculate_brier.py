#!/usr/bin/env python3
"""
Script to calculate Brier score for uncertainty quantification metrics across multiple runs.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
import scipy.stats as stats

# allow imports from project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import DataLoader
from src.utils.config_loader import load_yaml_with_imports


def load_config(config_path: Path) -> Dict[str, Any]:
    config = load_yaml_with_imports(config_path)
    datasets = config.get('datasets', [])
    data_dir = config.get('data_dir')
    results_path = config.get('results_path')
    model = config.get('model')
    runs = config.get('runs', [])
    # resolve metrics manually
    raw_metrics = config.get('metrics', [])
    metrics: List[str] = []
    for item in raw_metrics:
        if isinstance(item, str) and item.startswith('@'):
            group = item[1:]
            entries = config.get(group, [])
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict) and 'name' in e:
                        metrics.append(e['name'])
                    elif isinstance(e, str):
                        metrics.append(e)
        else:
            metrics.append(item)
    # remove duplicates while preserving order
    seen = set()
    metrics = [m for m in metrics if not (m in seen or seen.add(m))]
    return {
        'datasets': datasets,
        'data_dir': data_dir,
        'model': model,
        'results_path': results_path,
        'runs': runs,
        'metrics': metrics,
    }


def process_run(data_loader: DataLoader, run_id: int, dataset: str, metrics: List[str]) -> Dict[str, Any]:
    data = data_loader.load_run_data(run_id, dataset)
    if not data or 'examples' not in data:
        print(f"No data for run {run_id}, dataset {dataset}")
        return {}
    y_true = np.array([1 if ex.get('label') else 0 for ex in data['examples']])
    N = len(y_true)
    results: Dict[str, Any] = {}
    for metric in metrics:
        key = f"{metric}_confidences"
        if key not in data:
            print(f"Warning: Metric '{metric}' not in run {run_id},{dataset}")
            continue
        conf = np.array(data[key])
        brier = float(np.mean((conf - y_true) ** 2))
        results[metric] = {'score': brier, 'n_samples': N}
    return results


def aggregate_across_runs(all_run_results: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}
    scores_by_metric: Dict[str, List[float]] = {}
    total_samples: Dict[str, int] = {}
    for run_res in all_run_results:
        for m, vals in run_res.items():
            scores_by_metric.setdefault(m, []).append(vals['score'])
            total_samples[m] = total_samples.get(m, 0) + vals.get('n_samples', 0)
    for m, scores in scores_by_metric.items():
        arr = np.array(scores)
        n_runs = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n_runs > 1 else 0.0
        median = float(np.median(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if n_runs > 1:
            ci = stats.t.interval(0.95, n_runs-1, loc=mean, scale=stats.sem(arr))
            ci = [float(ci[0]), float(ci[1])]
        else:
            ci = [mean, mean]
        aggregated[m] = {
            'mean_score': mean,
            'std_score': std,
            'median_score': median,
            'min_score': min_val,
            'max_score': max_val,
            'confidence_interval': ci,
            'n_runs': n_runs,
            'n_samples': int(total_samples.get(m, 0)),
        }
    return aggregated


def print_summary(agg: Dict[str, Any], metrics: List[str]) -> None:
    print("\nSummary of Brier scores:")
    print("-"*80)
    print(f"{'Metric':<30} {'Mean':<8} {'Std':<8} {'95% CI':<25} {'Min':<8} {'Max':<8} {'Runs':<4}")
    print("-"*80)
    for m in metrics:
        if m not in agg:
            continue
        s = agg[m]
        ci = s.get('confidence_interval', [np.nan, np.nan])
        print(f"{m:<30} {s['mean_score']:>7.4f} {s['std_score']:>7.4f} [{ci[0]:.4f}, {ci[1]:.4f}] {s['min_score']:>7.4f} {s['max_score']:>7.4f} {s['n_runs']:>4}")


def main():
    base = Path(__file__).parent.parent.parent
    import argparse
    parser = argparse.ArgumentParser(description='Calculate Brier scores')
    parser.add_argument('--config', type=str, default='configs/brier_config.yaml', help='Path to config YAML')
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base / config_path
    cfg = load_config(config_path)
    data_path = (base / cfg['data_dir']).resolve()
    results_dir = (base / cfg['results_path']).resolve()
    data_loader = DataLoader({'data_path': data_path, 'model': cfg['model']})
    for dataset in cfg['datasets']:
        print(f"\nProcessing Brier dataset: {dataset}")
        ds_dir = results_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        all_results: List[Dict[str, Any]] = []
        for run in tqdm(cfg['runs'], desc=dataset):
            r = process_run(data_loader, run, dataset, cfg['metrics'])
            if r:
                all_results.append(r)
                od = ds_dir / f"run_{run}"
                od.mkdir(parents=True, exist_ok=True)
                with open(od / f"{dataset}_brier.json", 'w') as f:
                    json.dump(r, f, indent=2)
        if all_results:
            agg = aggregate_across_runs(all_results, cfg['metrics'])
            agg_dir = ds_dir / 'aggregated'
            agg_dir.mkdir(parents=True, exist_ok=True)
            with open(agg_dir / f"{dataset}_brier.json", 'w') as f:
                json.dump(agg, f, indent=2)
            print(f"Aggregated Brier saved to {agg_dir / f'{dataset}_brier.json'}")
            print_summary(agg, cfg['metrics'])
            md = f"# Brier Score Report for {dataset}\n\n"
            md += "| Metric | Mean | Std Dev | Min | Max | Runs |\n"
            md += "|--------|------:|--------:|----:|----:|-----:|\n"
            for m in cfg['metrics']:
                if m in agg:
                    s = agg[m]
                    md += f"| {m} | {s['mean_score']:.4f} | {s['std_score']:.4f} | {s['min_score']:.4f} | {s['max_score']:.4f} | {s['n_runs']} |\n"
            with open(ds_dir / f"{dataset}_brier_report.md", 'w') as f:
                f.write(md)
            print(f"Markdown report saved to {ds_dir / f'{dataset}_brier_report.md'}")


if __name__ == '__main__':
    main()
