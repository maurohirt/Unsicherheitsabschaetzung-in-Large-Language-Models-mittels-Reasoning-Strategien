#!/usr/bin/env python3
"""
Script to calculate Expected Calibration Error (ECE) for uncertainty quantification metrics across multiple runs.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

# allow imports from project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import DataLoader
from src.utils.config_loader import load_yaml_with_imports, resolve_metrics


def load_config(config_path: Path) -> Dict[str, Any]:
    config = load_yaml_with_imports(config_path)
    # dataset list
    datasets = config.get('datasets', [])
    # setup paths and parameters
    data_dir = config.get('data_dir')
    results_path = config.get('results_path')
    model = config.get('model')
    runs = config.get('runs', [])
    n_bins = config.get('n_bins', 10)
    # resolve metrics groups manually
    raw_metrics = config.get('metrics', [])
    metrics = []
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
        'data_path': data_dir,
        'model': model,
        'results_path': results_path,
        'runs': runs,
        'n_bins': n_bins,
        'metrics': metrics,
    }


def process_run(data_loader: DataLoader, run_id: int, dataset: str, metrics: List[str], n_bins: int) -> Dict[str, Any]:
    data = data_loader.load_run_data(run_id, dataset)
    if not data or 'examples' not in data:
        print(f"No data for run {run_id}, dataset {dataset}")
        return {}
    y_true = np.array([1 if ex.get('label') else 0 for ex in data['examples']])
    results: Dict[str, Any] = {}
    # compute for each metric
    for metric in metrics:
        key = f"{metric}_confidences"
        if key not in data:
            print(f"Warning: Metric '{metric}' not in run {run_id},{dataset}")
            continue
        conf = np.array(data[key])
        # define bins
        bins = np.linspace(0.0, 1.0, n_bins+1)
        # assign to bins
        bin_idx = np.digitize(conf, bins, right=True) - 1
        bin_acc = []
        bin_conf = []
        bin_count = []
        N = len(y_true)
        for i in range(n_bins):
            idxs = np.where(bin_idx == i)[0]
            count = len(idxs)
            bin_count.append(count)
            if count > 0:
                a = float(np.mean(y_true[idxs]))
                c = float(np.mean(conf[idxs]))
            else:
                a = float('nan'); c = float('nan')
            bin_acc.append(a)
            bin_conf.append(c)
        # expected calibration error
        ece = float(np.nansum([abs(bin_conf[i] - bin_acc[i]) * bin_count[i] for i in range(n_bins)]) / N)
        results[metric] = {
            'score': ece,
            'bin_accuracy': bin_acc,
            'bin_confidence': bin_conf,
            'bin_count': bin_count,
            'bin_edges': bins.tolist(),
            'n_samples': N,
        }
    return results


def aggregate_across_runs(all_run_results: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Any]:
    import scipy.stats as stats
    aggregated: Dict[str, Any] = {}
    for metric in metrics:
        # gather per-run metrics
        runs_data = [r[metric] for r in all_run_results if metric in r]
        if not runs_data:
            continue
        scores = [d['score'] for d in runs_data]
        n_runs = len(scores)
        total_samples = sum(d.get('n_samples', 0) for d in runs_data)
        mean_ece = float(np.mean(scores))
        std_ece = float(np.std(scores, ddof=1)) if n_runs > 1 else 0.0
        median_ece = float(np.median(scores))
        min_ece = float(np.min(scores))
        max_ece = float(np.max(scores))
        # 95% CI
        if n_runs > 1:
            ci = stats.t.interval(0.95, n_runs-1, loc=mean_ece, scale=stats.sem(scores))
            ci = [float(ci[0]), float(ci[1])]
        else:
            ci = [mean_ece, mean_ece]
        # aggregate bins
        n_bins = len(runs_data[0]['bin_accuracy'])
        bin_edges = runs_data[0]['bin_edges']
        acc_matrix = np.array([d['bin_accuracy'] for d in runs_data])
        conf_matrix = np.array([d['bin_confidence'] for d in runs_data])
        count_matrix = np.array([d['bin_count'] for d in runs_data])
        mean_bin_acc = np.nanmean(acc_matrix, axis=0).tolist()
        std_bin_acc = np.nanstd(acc_matrix, axis=0, ddof=1 if n_runs>1 else 0).tolist()
        mean_bin_conf = np.nanmean(conf_matrix, axis=0).tolist()
        std_bin_conf = np.nanstd(conf_matrix, axis=0, ddof=1 if n_runs>1 else 0).tolist()
        sum_bin_count = np.sum(count_matrix, axis=0).tolist()
        aggregated[metric] = {
            'mean_ece': mean_ece,
            'std_ece': std_ece,
            'median_ece': median_ece,
            'min_ece': min_ece,
            'max_ece': max_ece,
            'confidence_interval': ci,
            'n_runs': n_runs,
            'n_samples': total_samples,
            'bin_edges': bin_edges,
            'mean_bin_accuracy': mean_bin_acc,
            'std_bin_accuracy': std_bin_acc,
            'mean_bin_confidence': mean_bin_conf,
            'std_bin_confidence': std_bin_conf,
            'bin_count': sum_bin_count,
        }
    return aggregated


def print_summary(aggregated: Dict[str, Any], metrics: List[str]) -> None:
    print("\nSummary of ECE scores:")
    print("-"*80)
    print(f"{'Metric':<30} {'Mean':<10} {'Std':<10} {'95% CI':<25} {'Min':<8} {'Max':<8} {'Runs'}")
    print("-"*80)
    for m in metrics:
        if m not in aggregated:
            continue
        s = aggregated[m]
        ci = s.get('confidence_interval', [np.nan, np.nan])
        print(f"{m:<30} {s['mean_ece']:>7.4f}    {s['std_ece']:>7.4f}    [{ci[0]:.4f}, {ci[1]:.4f}]    {s['min_ece']:>7.4f}    {s['max_ece']:>7.4f}    {s['n_runs']}")


def main():
    base = Path(__file__).parent.parent.parent
    config_path = base / 'configs' / 'ece_config.yaml'
    config = load_config(config_path)
    # resolve absolute paths
    config['data_path'] = (base / config['data_path']).resolve()
    results_dir = (base / config['results_path']).resolve()
    # init
    data_loader = DataLoader({'data_path': config['data_path'], 'model': config['model']})
    datasets = config['datasets']
    metrics = config['metrics']
    runs = config['runs']
    n_bins = config['n_bins']
    print(f"Calculating ECE for datasets: {datasets}, metrics: {metrics}, runs: {runs}, bins: {n_bins}")
    # process each dataset
    for dataset in datasets:
        print(f"\nProcessing {dataset}")
        ds_dir = results_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        # per-run
        for run in tqdm(runs, desc=f"{dataset}"):
            r = process_run(data_loader, run, dataset, metrics, n_bins)
            if r:
                all_results.append(r)
                od = ds_dir / f"run_{run}"
                od.mkdir(parents=True, exist_ok=True)
                with open(od / f"{dataset}_ece.json", 'w') as f:
                    json.dump(r, f, indent=2)
        # aggregate
        if all_results:
            agg = aggregate_across_runs(all_results, metrics)
            agg_dir = ds_dir / 'aggregated'
            agg_dir.mkdir(parents=True, exist_ok=True)
            with open(agg_dir / f"{dataset}_ece.json", 'w') as f:
                json.dump(agg, f, indent=2)
            print(f"Aggregated ECE saved to {agg_dir / f'{dataset}_ece.json'}")
            print_summary(agg, metrics)
            # markdown
            md = f"# ECE Report for {dataset}\n\n"
            md += "## Aggregated Results\n\n"
            md += "| Metric | Mean | Std Dev | Min | Max | Runs |\n"
            md += "|--------|------:|--------:|----:|----:|-----:|\n"
            for m in metrics:
                if m in agg:
                    s = agg[m]
                    md += f"| {m} | {s['mean_ece']:.4f} | {s['std_ece']:.4f} | {s['min_ece']:.4f} | {s['max_ece']:.4f} | {s['n_runs']} |\n"
            with open(ds_dir / f"{dataset}_ece_report.md", 'w') as f:
                f.write(md)
            print(f"Markdown report saved to {ds_dir / f'{dataset}_ece_report.md'}")

if __name__ == '__main__':
    main()
