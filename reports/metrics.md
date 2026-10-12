# Evaluation

20 raw candidates per cell for each of the 96 cells, seeds 42, 43, 44. Values are mean ± standard deviation over seeds. Filtered means the candidates that pass all seven filters.

## Perplexity

| Model | Validation | Test |
| --- | --- | --- |
| LSTM | 3.81 | 4.02 |
| Trigram | 10.17 | 9.34 |

## Generated questions

| Metric | LSTM raw | LSTM filtered | Trigram raw | Trigram filtered | Real test |
| --- | --- | --- | --- | --- | --- |
| novelty | 0.371 ± 0.023 | 1.000 ± 0.000 | 0.606 ± 0.031 | 1.000 ± 0.000 | 1.000 |
| distinct_1 | 0.053 ± 0.000 | 0.156 ± 0.003 | 0.059 ± 0.001 | 0.127 ± 0.002 | 0.224 |
| distinct_2 | 0.220 ± 0.004 | 0.414 ± 0.010 | 0.232 ± 0.011 | 0.360 ± 0.010 | 0.513 |
| self_bleu | 0.760 ± 0.022 | 0.534 ± 0.030 | 0.730 ± 0.034 | 0.558 ± 0.015 | 0.452 |
| relevance | 0.251 ± 0.003 | 0.267 ± 0.001 | 0.270 ± 0.005 | 0.300 ± 0.004 | 0.242 |
| topic_accuracy | 0.799 ± 0.012 | 0.860 ± 0.020 | 0.925 ± 0.012 | 0.961 ± 0.003 | 0.846 |
| difficulty_accuracy | 0.999 ± 0.001 | 0.998 ± 0.002 | 0.728 ± 0.002 | 0.990 ± 0.005 | 0.991 |
| pass_rate | 0.224 ± 0.011 | 0.224 ± 0.011 | 0.284 ± 0.010 | 0.284 ± 0.010 | - |

Classifier ceiling on real test questions: topic 0.846, difficulty 0.991.

## Ablations (seed 42, raw output)

| Variant | Val perplexity | novelty | distinct_1 | distinct_2 | self_bleu | relevance | topic_accuracy | difficulty_accuracy | pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temperature 0.6 | 3.81 | 0.229 | 0.050 | 0.170 | 0.834 | 0.272 | 0.864 | 1.000 | 0.147 |
| temperature 0.8 | 3.81 | 0.357 | 0.053 | 0.214 | 0.788 | 0.247 | 0.783 | 0.998 | 0.216 |
| temperature 1.0 | 3.81 | 0.490 | 0.053 | 0.255 | 0.719 | 0.224 | 0.705 | 0.998 | 0.270 |
| hidden size 128 | 5.05 | 0.729 | 0.038 | 0.130 | 0.943 | 0.151 | 0.193 | 0.997 | 0.217 |
| hidden size 512 | 3.92 | 0.372 | 0.052 | 0.221 | 0.762 | 0.259 | 0.822 | 1.000 | 0.237 |
| dropout 0.4 | 3.82 | 0.362 | 0.052 | 0.203 | 0.799 | 0.240 | 0.726 | 1.000 | 0.218 |
| no control tokens | 5.93 | 0.250 | 0.005 | 0.007 | 1.000 | 0.110 | 0.031 | 0.333 | 0.013 |
