import os
import json
import argparse

from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve
from tot.models import gpt_usage

def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i) 
        else:
            ys, info = solve(args, task, i)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        # log main metric
        accs = [info['r'] for info in infos]
        if accs:
            cnt_avg += sum(accs) / len(accs)
            cnt_any += any(accs)
        else:
            # no candidate produced a complete answer: count as zero
            cnt_avg += 0
            cnt_any += 0
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', gpt_usage(args.backend))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'deepseek-chat', 'deepseek-reasoner'], default='deepseek-chat')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy', 'random'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--n_propose_sample', type=int, default=1, help='Number of single-solution propose calls when uq_metric is set')

    # Uncertainty metric; empty string keeps legacy behaviour
    args.add_argument('--uq_metric', type=str, default='', choices=['', 'mean', 'min', 'max', 'entropy', 'random'])

    # For Game24 propose-mode generation, choose between variants
    # - single: loop single-solution calls with token-UQ scoring
    # - multi: one multi-solution call, score each line via token-UQ
    # - heuristic: multi-solution call, scored by heuristic value evaluator (no UQ)
    args.add_argument('--propose_uq_style', type=str, default='single', choices=['single', 'multi', 'heuristic'])

    args = args.parse_args()

    # Normalize args for Game24 propose variants
    if args.task == 'game24' and args.method_generate == 'propose':
        if args.propose_uq_style == 'heuristic':
            # Disable UQ; ensure heuristic evaluator is used with multiple samples
            args.uq_metric = ''
            args.method_evaluate = 'value'
            # Use 3 evaluator samples as per original variant
            if args.n_evaluate_sample < 3:
                args.n_evaluate_sample = 3
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)