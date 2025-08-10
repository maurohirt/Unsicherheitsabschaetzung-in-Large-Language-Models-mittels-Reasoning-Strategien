# Game of 24 (Tree-of-Thought BFS)

This repository implements a Tree-of-Thought (ToT) breadth-first search (BFS) solver for the Game of 24. At each step the system:
- generates candidate next moves (propose),
- scores them (evaluate),
- keeps the best ones (select),
- repeats for a few steps, and finally outputs a full solution expression that equals 24.

This README documents only the Game of 24 workflow and the three propose variants you can choose via the CLI.

## Where the logic lives
- Propose/evaluate/select loop: `src/tot/methods/bfs.py` (see `solve()` and `get_proposals()`)
- Game24 task/prompt wrappers: `src/tot/tasks/game24.py`
- Game24 prompts: `src/tot/prompts/game24.py`
- CLI entrypoint and flags: `run.py`

## Propose variants
Choose one with `--propose_uq_style` when `--method_generate propose`.

1) single — single-solution per LLM call with token-level UQ
- Makes repeated single-solution propose calls until it collects the requested number of unique next steps.
- Scores each proposed line using token log-probabilities (metric set by `--uq_metric`).
- `--n_propose_sample` controls how many single-solution calls to collect per state.

2) multi — multi-solution per LLM call with token-level UQ
- One LLM call returns multiple “Possible next steps:” lines.
- Each line is scored via token log-probabilities (metric set by `--uq_metric`).

3) heuristic — multi-solution per LLM call with heuristic evaluator (no UQ)
- One LLM call returns multiple “Possible next steps:” lines.
- Candidates are scored by a heuristic value judge (sure/likely/impossible) and aggregated across multiple judge calls.
- `--n_evaluate_sample` is set to at least 3 for better stability in this mode.

## Common flags (Game of 24)
- `--task game24` — select the task
- `--method_generate propose` — use sequential proposal generation
- `--method_evaluate value` — use the heuristic evaluator (or token-UQ scoring when enabled internally)
- `--method_select greedy` — keep the top-K candidates each step
- `--n_select_sample` — beam size (how many candidates to keep)
- `--n_evaluate_sample` — number of heuristic evaluations (used in `heuristic` mode; can stay 1 in UQ modes)
- `--n_propose_sample` — only used for `propose_uq_style=single` (how many single-solution proposals to gather)
- `--backend` — model to use (e.g., `deepseek-chat`)

## Recommended temperatures and UQ settings
- Default temperature for all runs: `0.7`
- Exception: for multi-solution UQ (`--propose_uq_style multi`) use `--temperature 1.8` and `--uq_metric mean`.

## Setup
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Export your API credentials in the terminal before running commands:
  - DeepSeek (examples use `--backend deepseek-chat`):
    ```bash
    export OPENAI_API_KEY="<your_deepseek_api_key>"
    export OPENAI_API_BASE="https://api.deepseek.com"
    ```
  - OpenAI: (Does not support UQ-Variants no token probabilites)
    ```bash
    export OPENAI_API_KEY="<your_openai_api_key>"
    ```

## Quick start: example commands
Replace the index range as you like.

### A) Multi-solution per call with token-UQ (temperature 1.8, probas mean)
```bash
python run.py \
  --task game24 \
  --task_start_index 900 \
  --task_end_index 905 \
  --backend deepseek-chat \
  --method_generate propose \
  --method_evaluate value \
  --method_select greedy \
  --propose_uq_style multi \
  --uq_metric mean \
  --n_evaluate_sample 1 \
  --n_select_sample 5 \
  --temperature 1.8
```

### B) Single-solution per call with token-UQ (recommended temp 0.7)
```bash
python run.py \
  --task game24 \
  --task_start_index 900 \
  --task_end_index 905 \
  --backend deepseek-chat \
  --method_generate propose \
  --method_evaluate value \
  --method_select greedy \
  --propose_uq_style single \
  --uq_metric min \
  --n_propose_sample 5 \
  --n_evaluate_sample 1 \
  --n_select_sample 5 \
  --temperature 0.7
```


### C) Multi-solution per call with heuristic evaluator (no UQ, temp 0.7)
```bash
python run.py \
  --task game24 \
  --task_start_index 900 \
  --task_end_index 905 \
  --backend deepseek-chat \
  --method_generate propose \
  --method_evaluate value \
  --method_select greedy \
  --propose_uq_style heuristic \
  --n_evaluate_sample 3 \
  --n_select_sample 5 \
  --temperature 0.7
```

## Output and logging
- Results and intermediate steps are printed to the console.
- JSON logs are saved under `./logs/game24/` with a filename that encodes your key flags.

## Notes
- `--uq_metric` options: `mean`, `min`, `max`, `entropy`, `random`. For the multi-solution UQ variant in this README, we use `mean`.
- `--n_generate_sample` is irrelevant in `propose` mode and can be left at its default.
- Make sure your API credentials and base URL are configured for your chosen backend (e.g., DeepSeek via OpenAI-compatible API).
