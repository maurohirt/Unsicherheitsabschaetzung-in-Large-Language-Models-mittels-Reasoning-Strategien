import itertools
import numpy as np
import random
from functools import partial

# Utility: preserve insertion order while removing duplicates
def distinct(seq):
    """Return a list with duplicate strings removed, preserving order."""
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
from tot.models import gpt
from tot.uq_utils import split_token_probs_by_line, line_metric, extract_tokens_logps_offsets
from tot.prompts.game24 import single_solution_propose_prompt, multiple_solutions_propose_prompt
from tot.tasks.game24 import get_current_numbers

def get_value(task, x, y, n_evaluate_sample, uq_metric='', cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    # detect whether this is the final answer step
    last_line = y.strip().split('\n')[-1]
    is_last_step = 'left: ' not in last_line
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    want_logprobs = bool(uq_metric and not is_last_step)
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None, return_logprobs=want_logprobs)
    if uq_metric and not is_last_step:
        # value_outputs are raw choice objects

        # we assume n_evaluate_sample == 1 for now
        choice = value_outputs[0]
        txt = choice.message.content
        toks, lps, offs = extract_tokens_logps_offsets(txt, choice.logprobs)
        lines = split_token_probs_by_line(txt, toks, lps, offs)
        # we expect one line only
        value = line_metric(lines[0][1], uq_metric)
    else:
        value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, uq_metric='', cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            if uq_metric and y in uq_score_cache:
                value = uq_score_cache[y]
            else:
                value = get_value(task, x, y, n_evaluate_sample, uq_metric=uq_metric, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

# global cache used across generation & evaluation in a run
uq_score_cache = {}

def get_proposals(task, x, y, uq_metric='', n_propose_sample=1):
    from tot.tasks.game24 import get_current_numbers
    curr = get_current_numbers(y if y else x)
    # final answer generation when only one number remains
    remaining = curr.strip().split()
    if len(remaining) <= 1:
        prompt = task.cot_prompt_wrap(x, y)
        resp = gpt(prompt, n=1, stop=None)[0]
        ans = resp.message.content if hasattr(resp, 'message') else resp
        return [y + ans]
    # non-final step
    # when no UQ: batch multi-solution prompt
    if not uq_metric:
        prompt = multiple_solutions_propose_prompt.format(input=curr)
        resp = gpt(prompt, n=1, stop=None)[0]
        text = resp.message.content if hasattr(resp, 'message') else resp
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        return [y + ln + '\n' for ln in lines]
    # UQ mode: multiple single-solution calls with uniqueness
    results = []
    seen = set()
    seen_ops = set()  # operations already proposed this step
    attempts = 0
    max_attempts = n_propose_sample *2
    while len(results) < n_propose_sample and attempts < max_attempts:
        attempts += 1
        avoid_block = ''
        if seen_ops:
            avoid_block = 'Already proposed (do NOT repeat):\n' + '\n'.join(seen_ops) + '\n\n'
        prompt = avoid_block + single_solution_propose_prompt.format(input=curr)
        resp = gpt(prompt, n=1, stop=None, return_logprobs=True)[0]
        txt = resp.message.content if hasattr(resp, 'message') else resp
        toks, lps, offs = extract_tokens_logps_offsets(txt, resp.logprobs)
        score = line_metric(lps, uq_metric)
        ln = txt.strip().split('\n')[0]
        candidate = y + ln + '\n'
        if candidate not in seen:
            seen.add(candidate)
            seen_ops.add(ln)
            uq_score_cache[candidate] = score
            results.append(candidate)
    # fill with highest-scored duplicates if needed
    if len(results) < n_propose_sample:
        sorted_seen = sorted(seen, key=lambda c: uq_score_cache[c], reverse=True)
        while len(results) < n_propose_sample:
            best = sorted_seen[0]
            results.append(best)
    return results

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, uq_metric=''):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    # detect final answer step
    last_line = y.strip().split('\n')[-1]
    is_last_step = 'left: ' not in last_line
    want_lp = bool(uq_metric and not is_last_step)
    samples = gpt(prompt, n=n_generate_sample, stop=stop, return_logprobs=want_lp)
    results = []
    for resp in samples:
        if want_lp:
            txt = resp.message.content
            toks, lps, offs = extract_tokens_logps_offsets(txt, resp.logprobs)
            score = line_metric(lps, uq_metric)
            cand = y + txt
            uq_score_cache[cand] = score
            results.append(cand)
        else:
            results.append(y + resp)
    return results

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step], uq_metric=args.uq_metric) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y, uq_metric=args.uq_metric, n_propose_sample=args.n_propose_sample) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        # 1) keep only distinct candidates
        new_ys = distinct(new_ys)

        # 2) special rule for step 2 (third arithmetic move)
        if step == 2:
            new_ys = [c for c in new_ys if 'left: 24' in c.strip().split('\n')[-1]]
            values = [1.0] * len(new_ys)  # dummy uniform values for logging
            k = min(args.n_select_sample, len(new_ys))
            select_ids = list(range(k))   # preserve original ordering
        else:
            ids = list(range(len(new_ys)))
            # combined selection: random/sample/greedy
            if args.method_select == 'random':
                values = [1.0] * len(new_ys)
                k = min(args.n_select_sample, len(ids))
                select_ids = random.sample(ids, k)
            else:
                # evaluation
                if args.method_evaluate == 'vote':
                    values = get_votes(task, x, new_ys, args.n_evaluate_sample)
                elif args.method_evaluate == 'value':
                    values = get_values(task, x, new_ys, args.n_evaluate_sample, uq_metric=args.uq_metric)
                # selection (respect actual available count)
                k = min(args.n_select_sample, len(ids))
                if args.method_select == 'sample':
                    ps = np.array(values) / sum(values) if sum(values) else np.ones(len(values)) / len(values)
                    select_ids = np.random.choice(ids, size=k, p=ps).tolist()
                elif args.method_select == 'greedy':
                    select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:k]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print:
            if new_ys and values:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
            else:
                print('-- no candidates this step --\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}