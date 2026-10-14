# Evaluation

20 raw candidates per cell for each of the 96 cells, seeds 42, 43, 44. Values are mean ± standard deviation over seeds. Filtered means the candidates that pass all seven filters and the phrase check.

## Perplexity

| Model | Validation | Test |
| --- | --- | --- |
| LSTM | 3.84 | 4.05 |
| Trigram | 10.17 | 9.34 |

## Generated questions

| Metric | LSTM raw | LSTM filtered | Trigram raw | Trigram filtered | Real test |
| --- | --- | --- | --- | --- | --- |
| novelty | 0.377 ± 0.028 | 1.000 ± 0.000 | 0.606 ± 0.031 | 1.000 ± 0.000 | 1.000 |
| distinct_1 | 0.053 ± 0.001 | 0.207 ± 0.006 | 0.059 ± 0.001 | 0.138 ± 0.005 | 0.224 |
| distinct_2 | 0.208 ± 0.005 | 0.391 ± 0.015 | 0.232 ± 0.011 | 0.360 ± 0.012 | 0.513 |
| self_bleu | 0.777 ± 0.013 | 0.573 ± 0.024 | 0.730 ± 0.034 | 0.567 ± 0.015 | 0.452 |
| relevance | 0.242 ± 0.003 | 0.292 ± 0.001 | 0.270 ± 0.005 | 0.300 ± 0.004 | 0.242 |
| topic_accuracy | 0.746 ± 0.017 | 0.902 ± 0.006 | 0.925 ± 0.012 | 0.965 ± 0.002 | 0.846 |
| difficulty_accuracy | 0.997 ± 0.002 | 1.000 ± 0.000 | 0.728 ± 0.002 | 0.990 ± 0.004 | 0.991 |
| pass_rate | 0.117 ± 0.004 | 0.117 ± 0.004 | 0.252 ± 0.004 | 0.252 ± 0.004 | - |

Classifier ceiling on real test questions: topic 0.846, difficulty 0.991.

## Ablations (seed 42, raw output)

| Variant | Val perplexity | novelty | distinct_1 | distinct_2 | self_bleu | relevance | topic_accuracy | difficulty_accuracy | pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temperature 0.6 | 3.84 | 0.247 | 0.049 | 0.163 | 0.861 | 0.266 | 0.809 | 1.000 | 0.106 |
| temperature 0.8 | 3.84 | 0.351 | 0.052 | 0.201 | 0.795 | 0.238 | 0.723 | 0.998 | 0.115 |
| temperature 1.0 | 3.84 | 0.457 | 0.053 | 0.242 | 0.756 | 0.214 | 0.638 | 0.996 | 0.098 |
| hidden size 128 | 5.38 | 0.715 | 0.033 | 0.101 | 0.961 | 0.137 | 0.122 | 0.998 | 0.044 |
| hidden size 512 | 3.92 | 0.410 | 0.051 | 0.212 | 0.791 | 0.248 | 0.767 | 0.999 | 0.105 |
| dropout 0.4 | 3.86 | 0.376 | 0.051 | 0.192 | 0.826 | 0.226 | 0.640 | 1.000 | 0.108 |
| no control tokens | 5.93 | 0.250 | 0.005 | 0.008 | 1.000 | 0.104 | 0.031 | 0.333 | 0.009 |
