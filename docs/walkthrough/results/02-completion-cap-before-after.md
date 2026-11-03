# Quant failure suite - before and after

- before: `openrouter/cohere/north-mini-code:free`, completion cap 800 tokens, 2026-11-03
- after: `openrouter/cohere/north-mini-code:free`, completion cap 3000 tokens, 2026-11-03

| metric | before (n, 95% interval) | after | verdict |
|---|---|---|---|
| solve.mean_pass_rate | 0.83 (n=10, 0.60-1.00) | 0.80 | within noise |
| solve.pass_hat_k | 0.80 (n=10, 0.49-0.94) | 0.80 | within noise |
| tutor.mean_pass_rate | 0.89 (n=12, 0.81-0.97) | 0.83 | within noise |
| tutor.pass_hat_k | 0.67 (n=12, 0.39-0.86) | 0.67 | within noise |
| tutor_multiturn.mean_pass_rate | 0.33 (n=4, 0.08-0.58) | 0.25 | within noise |
| tutor_multiturn.pass_hat_k | 0.00 (n=4, 0.00-0.49) | 0.00 | within noise |
| security.pass_rate | 1.00 (n=8, 0.68-1.00) | 1.00 | within noise |

None of the 7 metrics moved outside the old interval: at this sample size the comparison cannot tell the two runs apart.
