| config | kernels | distinct | ops G | worst max diff | worst mean diff | step CPU ms (min of runs / median of runs) | step wall ms (min / median) | runs |
|---|---|---|---|---|---|---|---|---|
| variant:lean | 159 | 76 | 1.886 | 0.00053 | 8.9e-06 | 164 / 191 | 264 / 332 | 2 |
| variant:lean_foldnorm | 159 | 76 | 1.843 | 0.00079 | 1.5e-05 | 161 / 172 | 213 / 272 | 2 |
| variant:lean_foldnorm_foldscale | 159 | 76 | 1.843 | 0.00095 | 1.8e-05 | 160 / 175 | 233 / 264 | 2 |
| variant:lean_foldnorm_foldscale_concat | 148 | 92 | 1.843 | 0.00095 | 1.8e-05 | 156 / 173 | 205 / 263 | 2 |
| BEAM=1 (lean) | 159 | 76 | 1.984 | 0.00056 | 9.7e-06 | 140 / 219 | 173 / 275 | 1 |
