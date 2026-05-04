# SA Placer — Full Evaluation Results

**Algorithm:** Simulated Annealing (greedy shelf-pack init → SA 55 min/bench)  
**Hardware:** Local laptop (AMD, Windows 11)  
**Date run:** 2026-05-03 to 2026-05-04  

> Note: ibm03 shows 29,097s runtime due to laptop sleep mid-run. SA itself ran correctly — the score is valid.

---

## Per-Benchmark Results

| Benchmark | Proxy | WL | Density | Congestion | vs SA | vs RePlAce | Overlaps |
|-----------|-------|----|---------|------------|-------|------------|----------|
| ibm01 | 1.2967 | 0.065 | 1.054 | 1.409 | +1.5% | -30.0% | 0 |
| ibm02 | 1.8409 | 0.078 | 1.060 | 2.466 | +3.5% | -0.2% | 0 |
| ibm03 | 1.7131 | 0.090 | 1.028 | 2.218 | +1.6% | -29.6% | 0 |
| ibm04 | 1.6252 | 0.073 | 1.077 | 2.027 | -8.1% | -24.8% | 0 |
| ibm06 | 1.9216 | 0.064 | 1.021 | 2.694 | +23.3% | -18.7% | 0 |
| ibm07 | 1.7698 | 0.067 | 1.127 | 2.278 | +12.5% | -20.9% | 0 |
| ibm08 | 2.1493 | 0.080 | 1.159 | 2.980 | -11.7% | -50.5% | 0 |
| ibm09 | 1.4169 | 0.059 | 1.125 | 1.590 | -2.1% | -26.6% | 0 |
| ibm10 | 1.8698 | 0.076 | 1.013 | 2.574 | +11.4% | -24.6% | 0 |
| ibm11 | 1.4808 | 0.058 | 1.111 | 1.734 | +13.5% | -25.8% | 0 |
| ibm12 | 2.1812 | 0.077 | 1.038 | 3.170 | +22.8% | -26.4% | 0 |
| ibm13 | 1.6712 | 0.055 | 1.123 | 2.110 | +12.7% | -25.1% | 0 |
| ibm14 | 1.7050 | 0.051 | 1.110 | 2.198 | +25.1% | -10.5% | 0 |
| ibm15 | 1.6761 | 0.058 | 1.032 | 2.205 | +27.1% | -10.6% | 0 |
| ibm16 | 1.7028 | 0.049 | 1.016 | 2.292 | +23.8% | -15.2% | 0 |
| ibm17 | 1.8081 | 0.053 | 1.025 | 2.485 | +50.8% | -9.9% | 0 |
| ibm18 | 1.8296 | 0.053 | 1.107 | 2.447 | +34.1% | -3.2% | 0 |
| **AVG** | **1.7446** | | | | **+17.9%** | **-19.7%** | **0** |

---

## Summary

| | Score |
|---|---|
| Our average proxy | **1.7446** |
| SA baseline | 2.1251 |
| RePlAce baseline | 1.4578 |
| Beats SA by | 17.9% |
| vs RePlAce | 19.7% worse |
| Valid benchmarks | 17 / 17 |
| Total overlaps | 0 |

**Estimated leaderboard position:** ~rank 26-27 (between UT Austin RH 1.6037 and UT Austin CT 1.8706)

---

## Leaderboard Context (as of 2026-05-03)

| Rank | Team | Avg Proxy |
|------|------|-----------|
| 1 | Cezar (ReFine) | 1.2224 |
| 2 | MTK (DreamPlace++) | 1.2818 |
| 3 | RoRa (RipPlace) | 1.3241 |
| ... | | |
| 17 | another Waterloo kid (Batched Nesterov GP) | 1.4568 |
| — | RePlAce baseline | 1.4578 |
| ... | | |
| **~26** | **Ours (SA)** | **1.7446** |
| 27 | UT Austin CT (PROXYCost) | 1.8706 |
| — | SA baseline | 2.1251 |
