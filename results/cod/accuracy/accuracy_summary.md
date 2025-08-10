# Accuracy Results

## Per-Run Accuracy

| Run | Dataset | Accuracy | Correct/Total |
|-----|---------|---------:|---------------:|
| run_0 | 2WikimhQA | 0.0366 | 21/573 |
| run_0 | hotpotQA | 0.2064 | 1,534/7,432 |
| run_0 | gsm8k | 0.3766 | 496/1,317 |
| run_0 | svamp | 0.5870 | 587/1,000 |
| run_0 | ASDiv | 0.6566 | 1,474/2,245 |
| run_1 | 2WikimhQA | 0.0388 | 23/593 |
| run_1 | hotpotQA | 0.2152 | 1,612/7,491 |
| run_1 | gsm8k | 0.3728 | 491/1,317 |
| run_1 | svamp | 0.5660 | 566/1,000 |
| run_1 | ASDiv | 0.6656 | 1,497/2,249 |
| run_2 | 2WikimhQA | 0.0250 | 14/560 |
| run_2 | hotpotQA | 0.2092 | 1,560/7,456 |
| run_2 | gsm8k | 0.3784 | 498/1,316 |
| run_2 | svamp | 0.5560 | 556/1,000 |
| run_2 | ASDiv | 0.6658 | 1,496/2,247 |
| run_3 | 2WikimhQA | 0.0344 | 18/523 |
| run_3 | hotpotQA | 0.2098 | 1,563/7,451 |
| run_3 | gsm8k | 0.3799 | 500/1,316 |
| run_3 | svamp | 0.5740 | 574/1,000 |
| run_3 | ASDiv | 0.6698 | 1,505/2,247 |
| run_4 | 2WikimhQA | 0.0268 | 15/559 |
| run_4 | hotpotQA | 0.2104 | 1,300/6,178 |
| run_4 | gsm8k | 0.3877 | 511/1,318 |
| run_4 | svamp | 0.5536 | 553/999 |
| run_4 | ASDiv | 0.6566 | 1,476/2,248 |

## Aggregated by Dataset

| Dataset | Mean | Std Dev | Min | Max | Runs |
|---------|-----:|--------:|----:|----:|-----:|
| 2WikimhQA | 0.0323 | 0.0061 | 0.0250 | 0.0388 | 5 |
| hotpotQA | 0.2102 | 0.0032 | 0.2064 | 0.2152 | 5 |
| gsm8k | 0.3791 | 0.0055 | 0.3728 | 0.3877 | 5 |
| svamp | 0.5673 | 0.0137 | 0.5536 | 0.5870 | 5 |
| ASDiv | 0.6629 | 0.0060 | 0.6566 | 0.6698 | 5 |

## Overall Summary

- **Mean Accuracy**: 0.3316
- **Std Dev**: 0.2348
- **Range**: 0.0250 - 0.6698
- **Total Correct**: 20,440
- **Total Examples**: 61,635
- **Overall Accuracy**: 0.3316 (20,440/61,635)
