# Accuracy Results

## Per-Run Accuracy

| Run | Dataset | Accuracy | Correct/Total |
|-----|---------|---------:|---------------:|
| run_0 | 2WikimhQA | 0.0575 | 89/1,548 |
| run_0 | hotpotQA | 0.2752 | 2,311/8,397 |
| run_0 | gsm8k | 0.4146 | 510/1,230 |
| run_0 | svamp | 0.6110 | 611/1,000 |
| run_0 | ASDiv | 0.6471 | 1,441/2,227 |
| run_1 | 2WikimhQA | 0.0621 | 96/1,546 |
| run_1 | hotpotQA | 0.2725 | 2,286/8,390 |
| run_1 | gsm8k | 0.4063 | 505/1,243 |
| run_1 | svamp | 0.6080 | 608/1,000 |
| run_1 | ASDiv | 0.6542 | 1,457/2,227 |
| run_2 | 2WikimhQA | 0.0563 | 87/1,544 |
| run_2 | hotpotQA | 0.2687 | 2,258/8,404 |
| run_2 | gsm8k | 0.4062 | 499/1,231 |
| run_2 | svamp | 0.6022 | 601/998 |
| run_2 | ASDiv | 0.6350 | 1,418/2,233 |
| run_3 | 2WikimhQA | 0.0596 | 30/503 |
| run_3 | hotpotQA | 0.2450 | 1,709/6,975 |
| run_3 | gsm8k | 0.4261 | 507/1,190 |
| run_3 | svamp | 0.6054 | 603/996 |
| run_3 | ASDiv | 0.6391 | 1,413/2,211 |
| run_4 | 2WikimhQA | 0.0695 | 34/489 |
| run_4 | hotpotQA | 0.2424 | 1,139/4,698 |
| run_4 | gsm8k | 0.4420 | 514/1,163 |
| run_4 | svamp | 0.6099 | 605/992 |
| run_4 | ASDiv | 0.6540 | 1,450/2,217 |

## Aggregated by Dataset

| Dataset | Mean | Std Dev | Min | Max | Runs |
|---------|-----:|--------:|----:|----:|-----:|
| 2WikimhQA | 0.0610 | 0.0052 | 0.0563 | 0.0695 | 5 |
| hotpotQA | 0.2608 | 0.0157 | 0.2424 | 0.2752 | 5 |
| gsm8k | 0.4190 | 0.0152 | 0.4062 | 0.4420 | 5 |
| svamp | 0.6073 | 0.0035 | 0.6022 | 0.6110 | 5 |
| ASDiv | 0.6459 | 0.0087 | 0.6350 | 0.6542 | 5 |

## Overall Summary

- **Mean Accuracy**: 0.3524
- **Std Dev**: 0.2229
- **Range**: 0.0563 - 0.6542
- **Total Correct**: 22,781
- **Total Examples**: 64,652
- **Overall Accuracy**: 0.3524 (22,781/64,652)
