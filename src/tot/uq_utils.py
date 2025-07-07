import math
from typing import List, Tuple
import random

def split_token_probs_by_line(text: str, tokens: List[str], logps: List[float], offsets: List[int]) -> List[Tuple[str, List[float]]]:
    """Bucket token log-probs into their corresponding output lines.
    Returns list of (line_str, [token_logps]) excluding blank lines."""
    # Compute start indices of each line in the full text
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)
    line_starts.append(len(text) + 1)  # sentinel

    def offset_to_line(off: int) -> int:
        # binary search
        lo, hi = 0, len(line_starts) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if off >= line_starts[mid]:
                lo = mid
            else:
                hi = mid
        return lo

    buckets: List[Tuple[str, List[float]]] = [("", []) for _ in range(len(line_starts) - 1)]
    for tok, lp, off in zip(tokens, logps, offsets):
        lid = offset_to_line(off)
        buckets[lid] = (buckets[lid][0] + tok, buckets[lid][1] + [lp])

    return [(s.strip(), lps) for s, lps in buckets if s.strip()]


def extract_tokens_logps_offsets(text: str, lp_dict: dict):
    """Return (tokens, logps, offsets) lists for DeepSeek-style logprobs.
    DeepSeek returns {'content': [{'token': str, 'logprob': float, ...}, ...]}.
    It provides no offsets, so we reconstruct them by scanning the text.
    """
    if not lp_dict or 'content' not in lp_dict:
        return [], [], []
    entries = lp_dict['content']
    tokens = [e['token'] for e in entries]
    logps  = [e['logprob'] for e in entries]
    # rebuild offsets sequentially
    offsets = []
    pos = 0
    for tok in tokens:
        # advance until match (should align sequentially)
        idx = text.find(tok, pos)
        if idx == -1:
            idx = pos  # fallback
        offsets.append(idx)
        pos = idx + len(tok)
    return tokens, logps, offsets


def line_metric(logps: List[float], metric: str) -> float:
    if metric == "random":
        return random.random()
    if not logps:
        return 0.0
    if metric == "mean":
        probs = [math.exp(lp) for lp in logps]
        return sum(probs) / len(probs)
    if metric == "min":
        return math.exp(min(logps))
    if metric == "max":
        return math.exp(max(logps))
    if metric == "entropy":
        p = [math.exp(lp) for lp in logps]
        z = sum(p)
        p = [v / z for v in p]
        H = -sum(pi * math.log(pi + 1e-9) for pi in p)
        n = len(p)
        # normalize to [0,1] by dividing by log(n)
        if n > 1:
            return H / math.log(n)
        else:
            return 0.0
    raise ValueError(metric)
