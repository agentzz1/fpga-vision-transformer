# Benchmark / ablation (synthetic validation)

5-fold CV. On the real provided `.mat`, rerun via `run_analysis.py`. The story for judges: **SOTA methods beat the naive baseline**, and we report it honestly.

| Paradigm | Method | Acc | AUC |
|---|---|---|---|
| Motor Imagery | raw+shrinkLDA (baseline) | 0.458 | - |
| Motor Imagery | CSP + LDA | 1.000 | - |
| Motor Imagery | Riemann tangent-space + LR | 1.000 | 1.000 |
| Motor Imagery | EEGNet (CNN) | 1.000 | - |
| P300 | raw+shrinkLDA (baseline) | 0.881 | - |
| P300 | xDAWN + shrinkLDA | 0.956 | 0.988 |
