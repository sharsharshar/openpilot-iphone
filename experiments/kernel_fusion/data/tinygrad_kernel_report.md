161 launches, time column seq_min_us: sum 139.0 ms, ops 1.836 G, overall 13.2 GOPS

### Top 15 launches by time
| # | launch | stage / layer | ops (M) | us | share | GOPS |
|---|---|---|---|---|---|---|
| 1 | #5 r_16_64_32_4_4_24_3_3 | encoder stem / stem.0 | 249.6 | 16031 | 11.5% | 15.6 |
| 2 | #147 r_3_128_4_3_512_4 | temporal transformer / p_val_63 | 18.9 | 7206 | 5.2% | 2.6 |
| 3 | #146 r_3_512_4_3_128_4 | temporal transformer / p_val_61 | 19.7 | 6351 | 4.6% | 3.1 |
| 4 | #17 r_64_512_4_3_16_4 | encoder stage 0 / blocks.1 mlp.fc1 | 67.6 | 5458 | 3.9% | 12.4 |
| 5 | #13 r_64_512_4_3_16_4 | encoder stage 0 / blocks.0 mlp.fc1 | 67.6 | 5420 | 3.9% | 12.5 |
| 6 | #23 r_128_128_4_3_32_4 | encoder stage 1 / blocks.0 mlp.fc1 | 59.0 | 3818 | 2.7% | 15.4 |
| 7 | #27 r_128_128_4_3_32_4 | encoder stage 1 / blocks.1 mlp.fc1 | 59.0 | 3756 | 2.7% | 15.7 |
| 8 | #18 r_16_512_4_4_48_4 | encoder stage 0 / blocks.1 mlp.fc2+scale+res | 50.7 | 3372 | 2.4% | 15.0 |
| 9 | #14 r_16_512_4_4_48_4 | encoder stage 0 / blocks.0 mlp.fc2+scale+res | 50.7 | 3301 | 2.4% | 15.4 |
| 10 | #53 r_256_32_4_3_64_4 | encoder stage 2 / blocks.5 mlp.fc1 | 54.7 | 3210 | 2.3% | 17.0 |
| 11 | #45 r_256_32_4_3_64_4 | encoder stage 2 / blocks.3 mlp.fc1 | 54.7 | 3201 | 2.3% | 17.1 |
| 12 | #59 r_512_8_4_3_128_4 | encoder stage 3 / blocks.0 mlp.fc1 | 52.5 | 3179 | 2.3% | 16.5 |
| 13 | #33 r_256_32_4_3_64_4 | encoder stage 2 / blocks.0 mlp.fc1 | 54.7 | 3164 | 2.3% | 17.3 |
| 14 | #41 r_256_32_4_3_64_4 | encoder stage 2 / blocks.2 mlp.fc1 | 54.7 | 3164 | 2.3% | 17.3 |
| 15 | #37 r_256_32_4_3_64_4 | encoder stage 2 / blocks.1 mlp.fc1 | 54.7 | 3155 | 2.3% | 17.3 |

### By stage
| stage | launches | ops (M) | ops share | time ms | time share | GOPS |
|---|---|---|---|---|---|---|
| encoder stage 2 | 26 | 677.9 | 36.9% | 33.85 | 24.4% | 20.0 |
| encoder stage 0 | 8 | 270.7 | 14.7% | 20.06 | 14.4% | 13.5 |
| encoder stem | 3 | 280.2 | 15.3% | 18.84 | 13.6% | 14.9 |
| temporal transformer | 12 | 68.4 | 3.7% | 17.86 | 12.9% | 3.8 |
| per-frame heads (policy, no_bottleneck, summarizer) | 57 | 30.1 | 1.6% | 16.81 | 12.1% | 1.8 |
| encoder stage 1 | 10 | 263.1 | 14.3% | 14.47 | 10.4% | 18.2 |
| encoder stage 3 | 10 | 230.9 | 12.6% | 12.14 | 8.7% | 19.0 |
| temporal heads | 11 | 6.3 | 0.3% | 2.88 | 2.1% | 2.2 |
| encoder final conv+SE+pool+head | 4 | 5.0 | 0.3% | 1.09 | 0.8% | 4.6 |
| no weights (state/glue/softmax/etc.) | 19 | 1.8 | 0.1% | 0.70 | 0.5% | 2.6 |
| input normalisation / image prep | 1 | 1.6 | 0.1% | 0.29 | 0.2% | 5.4 |

### By kernel class
| cls | launches | ops (M) | ops share | time ms | time share | GOPS |
|---|---|---|---|---|---|---|
| conv 1x1 (fc1, fc2, downsample, stem.2) | 31 | 1374.4 | 74.9% | 75.47 | 54.3% | 18.2 |
| linear layers, M=1 (matvec) | 72 | 37.7 | 2.1% | 20.55 | 14.8% | 1.8 |
| temporal matmul (token-batched) | 6 | 67.1 | 3.7% | 17.00 | 12.2% | 3.9 |
| conv dense 3x3 (stem.0) | 1 | 249.6 | 13.6% | 16.03 | 11.5% | 15.6 |
| conv depthwise 7x7 (+fused epilogue) | 15 | 83.1 | 4.5% | 7.14 | 5.1% | 11.6 |
| conv depthwise 3x3 (+fused epilogue) | 14 | 20.6 | 1.1% | 1.82 | 1.3% | 11.3 |
| elementwise E_* | 9 | 1.6 | 0.1% | 0.65 | 0.5% | 2.5 |
| other reduce r_* (norm/softmax/pool/etc.) | 13 | 1.8 | 0.1% | 0.35 | 0.2% | 5.2 |

### Launch size buckets (time column)
| launch time | launches | time ms | time share | fixed-cost share if each launch costs 15 us extra |
|---|---|---|---|---|
| 0-10 us | 39 | 0.20 | 0.1% | 75% |
| 10-50 us | 24 | 0.70 | 0.5% | 34% |
| 50-200 us | 10 | 0.94 | 0.7% | 14% |
| 200-1000 us | 52 | 24.36 | 17.5% | 3% |
| 1000-inf us | 36 | 112.81 | 81.2% | 0% |

launch cost 15 us x 161 = 2.42 ms = 1.7% of (kernel time + launch cost)

OLS fit, all launches with ops>0: time = 190.9 us + 0.059 ns/op  (= 16.9 GOPS asymptotic); R^2 = 0.812
OLS fit, dense + 1x1 conv launches only (n=32): time = -418.7 us + 0.065 ns/op (15.5 GOPS)
launches with < 3000 ops (n=24): median time 3.48 us, min 2.40 us (this is the harness cost of one direct call: ctypes + timer, NOT the jitted launch cost)
GOPS over launches with > 0.1 Mop: min 1.3, median 10.9, max 26.1
