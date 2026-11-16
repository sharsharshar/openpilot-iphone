model models/driving_supercombo_fp32.onnx: 377 nodes, 847.0 M MACs (= 1.694 G ops at 2 ops/MAC)
ORT 1 thread: sum of per-node median kernel time (opt off) 46.5 ms; profiled whole-run median 50.5 ms
ORT unprofiled whole run ORT_DISABLE_ALL: median 46.6 ms, min 41.5 ms (n=30)
ORT unprofiled whole run ORT_ENABLE_ALL: median 31.4 ms, min 28.3 ms (n=30)
nodes with median kernel time < 5 us (timer resolution is 1 us): 80

### Share per stage
| stage | nodes | MMACs | MAC share | time ms | time share | MB out |
|---|---|---|---|---|---|---|
| encoder stage 2 | 45 | 323.4 | 38.2% | 11.94 | 25.7% | 9.04 |
| encoder stage 0 | 14 | 115.9 | 13.7% | 6.97 | 15.0% | 11.53 |
| per-frame heads (vision policy + no_bottleneck + summarizer) | 149 | 16.5 | 2.0% | 6.09 | 13.1% | 0.20 |
| encoder stem | 6 | 122.8 | 14.5% | 5.36 | 11.5% | 6.29 |
| encoder stage 1 | 17 | 119.9 | 14.2% | 5.27 | 11.3% | 6.55 |
| encoder stage 3 | 17 | 111.8 | 13.2% | 4.27 | 9.2% | 1.64 |
| temporal transformer | 52 | 33.7 | 4.0% | 2.36 | 5.1% | 1.06 |
| state/input prep (slice, gather, cast, concat) | 24 | 0.0 | 0.0% | 2.03 | 4.4% | 36.34 |
| temporal heads + output concat | 31 | 2.6 | 0.3% | 0.96 | 2.1% | 0.07 |
| input normalisation (Sub, Div) | 2 | 0.0 | 0.0% | 0.74 | 1.6% | 6.29 |
| encoder final conv+SE+pool | 10 | 0.4 | 0.1% | 0.35 | 0.7% | 0.41 |
| state carry (slice/concat) | 4 | 0.0 | 0.0% | 0.14 | 0.3% | 0.40 |
| glue (slice/reshape/concat) | 6 | 0.0 | 0.0% | 0.04 | 0.1% | 0.06 |

### Share per op type
| op | nodes | MMACs | MAC share | time ms | time share | MB out |
|---|---|---|---|---|---|---|
| Conv | 60 | 794.1 | 93.8% | 31.31 | 67.3% | 19.14 |
| Gemm | 82 | 19.8 | 2.3% | 6.74 | 14.5% | 0.12 |
| Gelu | 20 | 0.0 | 0.0% | 2.26 | 4.9% | 11.28 |
| MatMul | 8 | 33.1 | 3.9% | 1.84 | 4.0% | 0.22 |
| Concat | 8 | 0.0 | 0.0% | 0.91 | 2.0% | 11.28 |
| Cast | 7 | 0.0 | 0.0% | 0.55 | 1.2% | 3.21 |
| Slice | 11 | 0.0 | 0.0% | 0.51 | 1.1% | 9.74 |
| Add | 47 | 0.0 | 0.0% | 0.46 | 1.0% | 2.81 |
| Sub | 1 | 0.0 | 0.0% | 0.42 | 0.9% | 3.15 |
| Mul | 19 | 0.0 | 0.0% | 0.37 | 0.8% | 2.65 |
| Div | 3 | 0.0 | 0.0% | 0.33 | 0.7% | 3.15 |
| Relu | 64 | 0.0 | 0.0% | 0.26 | 0.6% | 0.09 |
| Gather | 3 | 0.0 | 0.0% | 0.23 | 0.5% | 7.87 |
| Reshape | 9 | 0.0 | 0.0% | 0.05 | 0.1% | 3.23 |
| Transpose | 7 | 0.0 | 0.0% | 0.04 | 0.1% | 0.21 |
| ReduceMean | 2 | 0.0 | 0.0% | 0.03 | 0.1% | 0.01 |
| LayerNormalization | 2 | 0.0 | 0.0% | 0.03 | 0.1% | 0.04 |
| ReduceL2 | 2 | 0.0 | 0.0% | 0.03 | 0.1% | 0.00 |
| Unsqueeze | 4 | 0.0 | 0.0% | 0.03 | 0.1% | 1.58 |
| Sigmoid | 4 | 0.0 | 0.0% | 0.02 | 0.0% | 0.03 |
| Where | 1 | 0.0 | 0.0% | 0.02 | 0.0% | 0.00 |
| Expand | 2 | 0.0 | 0.0% | 0.02 | 0.0% | 0.00 |
| Softmax | 1 | 0.0 | 0.0% | 0.02 | 0.0% | 0.00 |
| Clip | 2 | 0.0 | 0.0% | 0.01 | 0.0% | 0.00 |
| Identity | 3 | 0.0 | 0.0% | 0.01 | 0.0% | 0.01 |
| Squeeze | 3 | 0.0 | 0.0% | 0.01 | 0.0% | 0.06 |
| ReduceMax | 1 | 0.0 | 0.0% | 0.01 | 0.0% | 0.00 |
| GatherND | 1 | 0.0 | 0.0% | 0.01 | 0.0% | 0.02 |

### Share per op type, convolutions split by kind
| kind | nodes | MMACs | MAC share | time ms | time share | MB out |
|---|---|---|---|---|---|---|
| Conv 1x1 | 30 | 637.7 | 75.3% | 12.72 | 27.3% | 10.95 |
| Conv depthwise 7x7 | 12 | 30.5 | 3.6% | 10.52 | 22.6% | 2.49 |
| Gemm | 82 | 19.8 | 2.3% | 6.74 | 14.5% | 0.12 |
| Conv dense 3x3 | 1 | 113.2 | 13.4% | 3.67 | 7.9% | 2.10 |
| Conv depthwise 3x3 | 13 | 6.8 | 0.8% | 2.83 | 6.1% | 3.01 |
| Gelu | 20 | 0.0 | 0.0% | 2.26 | 4.9% | 11.28 |
| MatMul | 8 | 33.1 | 3.9% | 1.84 | 4.0% | 0.22 |
| Conv grouped 7x7 | 3 | 5.6 | 0.7% | 1.37 | 2.9% | 0.46 |
| Concat | 8 | 0.0 | 0.0% | 0.91 | 2.0% | 11.28 |
| Cast | 7 | 0.0 | 0.0% | 0.55 | 1.2% | 3.21 |
| Slice | 11 | 0.0 | 0.0% | 0.51 | 1.1% | 9.74 |
| Add | 47 | 0.0 | 0.0% | 0.46 | 1.0% | 2.81 |
| Sub | 1 | 0.0 | 0.0% | 0.42 | 0.9% | 3.15 |
| Mul | 19 | 0.0 | 0.0% | 0.37 | 0.8% | 2.65 |
| Div | 3 | 0.0 | 0.0% | 0.33 | 0.7% | 3.15 |
| Relu | 64 | 0.0 | 0.0% | 0.26 | 0.6% | 0.09 |
| Gather | 3 | 0.0 | 0.0% | 0.23 | 0.5% | 7.87 |
| Conv grouped 3x3 | 1 | 0.3 | 0.0% | 0.20 | 0.4% | 0.13 |
| Reshape | 9 | 0.0 | 0.0% | 0.05 | 0.1% | 3.23 |
| Transpose | 7 | 0.0 | 0.0% | 0.04 | 0.1% | 0.21 |
| ReduceMean | 2 | 0.0 | 0.0% | 0.03 | 0.1% | 0.01 |
| LayerNormalization | 2 | 0.0 | 0.0% | 0.03 | 0.1% | 0.04 |
| ReduceL2 | 2 | 0.0 | 0.0% | 0.03 | 0.1% | 0.00 |
| Unsqueeze | 4 | 0.0 | 0.0% | 0.03 | 0.1% | 1.58 |
| Sigmoid | 4 | 0.0 | 0.0% | 0.02 | 0.0% | 0.03 |
| Where | 1 | 0.0 | 0.0% | 0.02 | 0.0% | 0.00 |
| Expand | 2 | 0.0 | 0.0% | 0.02 | 0.0% | 0.00 |
| Softmax | 1 | 0.0 | 0.0% | 0.02 | 0.0% | 0.00 |
| Clip | 2 | 0.0 | 0.0% | 0.01 | 0.0% | 0.00 |
| Identity | 3 | 0.0 | 0.0% | 0.01 | 0.0% | 0.01 |
| Squeeze | 3 | 0.0 | 0.0% | 0.01 | 0.0% | 0.06 |
| ReduceMax | 1 | 0.0 | 0.0% | 0.01 | 0.0% | 0.00 |
| GatherND | 1 | 0.0 | 0.0% | 0.01 | 0.0% | 0.02 |

### Top 15 nodes by measured time
| # | idx | node | op | out shape | MMACs | ms | time share | cum |
|---|---|---|---|---|---|---|---|---|
| 1 | 26 | node_conv2d | Conv | 1x64x64x128 | 113.2 | 3.67 | 7.9% | 7.9% |
| 2 | 33 | node_Conv_173 | Conv | 1x64x32x64 | 6.4 | 1.59 | 3.4% | 11.3% |
| 3 | 40 | node_Conv_175 | Conv | 1x64x32x64 | 6.4 | 1.48 | 3.2% | 14.5% |
| 4 | 57 | node_Conv_179 | Conv | 1x128x16x32 | 3.2 | 0.94 | 2.0% | 16.5% |
| 5 | 50 | node_Conv_177 | Conv | 1x128x16x32 | 3.2 | 0.94 | 2.0% | 18.5% |
| 6 | 28 | node_conv2d_1 | Conv | 1x64x32x64 | 1.2 | 0.85 | 1.8% | 20.3% |
| 7 | 74 | node_Conv_183 | Conv | 1x256x8x16 | 1.6 | 0.74 | 1.6% | 21.9% |
| 8 | 67 | node_Conv_181 | Conv | 1x256x8x16 | 1.6 | 0.73 | 1.6% | 23.5% |
| 9 | 88 | node_Conv_187 | Conv | 1x256x8x16 | 1.6 | 0.73 | 1.6% | 25.1% |
| 10 | 81 | node_Conv_185 | Conv | 1x256x8x16 | 1.6 | 0.73 | 1.6% | 26.6% |
| 11 | 95 | node_Conv_189 | Conv | 1x256x8x16 | 1.6 | 0.73 | 1.6% | 28.2% |
| 12 | 102 | node_Conv_191 | Conv | 1x256x8x16 | 1.6 | 0.73 | 1.6% | 29.8% |
| 13 | 135 | node_linear | Gemm | 1x2048 | 2.1 | 0.72 | 1.5% | 31.3% |
| 14 | 36 | node_conv2d_6 | Conv | 1x64x32x64 | 25.2 | 0.72 | 1.5% | 32.9% |
| 15 | 46 | node_conv2d_11 | Conv | 1x128x16x32 | 3.2 | 0.66 | 1.4% | 34.3% |

### Top 15 nodes by MACs
| # | idx | node | MMACs | MAC share | ms | time share |
|---|---|---|---|---|---|---|
| 1 | 26 | node_conv2d | 113.2 | 13.4% | 3.67 | 7.9% |
| 2 | 34 | node_conv2d_5 | 25.2 | 3.0% | 0.51 | 1.1% |
| 3 | 36 | node_conv2d_6 | 25.2 | 3.0% | 0.72 | 1.5% |
| 4 | 41 | node_conv2d_9 | 25.2 | 3.0% | 0.52 | 1.1% |
| 5 | 43 | node_conv2d_10 | 25.2 | 3.0% | 0.66 | 1.4% |
| 6 | 51 | node_conv2d_15 | 25.2 | 3.0% | 0.44 | 0.9% |
| 7 | 53 | node_conv2d_16 | 25.2 | 3.0% | 0.47 | 1.0% |
| 8 | 58 | node_conv2d_19 | 25.2 | 3.0% | 0.44 | 1.0% |
| 9 | 60 | node_conv2d_20 | 25.2 | 3.0% | 0.48 | 1.0% |
| 10 | 68 | node_conv2d_25 | 25.2 | 3.0% | 0.47 | 1.0% |
| 11 | 70 | node_conv2d_26 | 25.2 | 3.0% | 0.45 | 1.0% |
| 12 | 75 | node_conv2d_29 | 25.2 | 3.0% | 0.47 | 1.0% |
| 13 | 77 | node_conv2d_30 | 25.2 | 3.0% | 0.46 | 1.0% |
| 14 | 82 | node_conv2d_33 | 25.2 | 3.0% | 0.46 | 1.0% |
| 15 | 84 | node_conv2d_34 | 25.2 | 3.0% | 0.45 | 1.0% |

### Concentration
top 3 nodes: 14.5% of time; top 3 by MACs: 19.3% of MACs
top 5 nodes: 18.5% of time; top 5 by MACs: 25.3% of MACs
top 10 nodes: 26.6% of time; top 10 by MACs: 40.1% of MACs
top 20 nodes: 40.8% of time; top 20 by MACs: 69.8% of MACs
first 3 convolutions (graph order): 14.5% of MACs, 10.1% of time
first 6 convolutions (graph order): 18.4% of MACs, 15.2% of time
first 10 convolutions (graph order): 25.2% of MACs, 21.7% of time
all encoder: 93.8% MACs, 73.4% time
nodes with zero MACs (data movement / elementwise / norm / softmax): 227 nodes, 14.3% of time
nodes needed for 90% of time: 97

First convolutions in graph order:
  idx 26 model/node_conv2d out 1x64x64x128 113.2 MMACs 3.67 ms
  idx 28 model/node_conv2d_1 out 1x64x32x64 1.2 MMACs 0.85 ms
  idx 30 model/node_conv2d_2 out 1x64x32x64 8.4 MMACs 0.18 ms
  idx 32 model/node_conv2d_3 out 1x64x32x64 1.2 MMACs 0.29 ms
  idx 33 model/node_Conv_173 out 1x64x32x64 6.4 MMACs 1.59 ms
  idx 34 model/node_conv2d_5 out 1x192x32x64 25.2 MMACs 0.51 ms
