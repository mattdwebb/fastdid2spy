# CPS did2s benchmark

Same frozen replication 1, 209,078 observations, 50 states, 2 early / 2 late / 46 never, eight years, no covariates, effect 1. The official comparison is an additional calculation on this same data, not another Monte Carlo draw.

| engine | representation | n | setup_seconds | point_seconds | native_seconds | jackknife_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| fast | raw_rows | 209078 | 0.168075 | 0.0032095 | 0.0022201 | 0.0040746 |
| reference | cell_WLS | 400 | 0.0007232 | 0.0008775 | 0.000844 | 0.029268 |

The raw explicit dummy oracle is intentionally size-bounded. For this no-covariate CPS specification only, 400 state-period means with cell-count WLS weights are an exact representation of the point estimate and state-cluster sums. This yields a transparent small-matrix oracle; it is not used to claim support for collapsing arbitrary individual covariates to means. Maximum raw-fast/cell-reference point error 5.11e-15, SE error 3.11e-16, and deletion error 1.73e-14.

Official R complete estimate + native covariance: 0.500000 s; official Stata: 1.422000 s. These include their model construction and are not directly comparable to a cached point-only call. R/Python point discrepancy 1.03273e-12, SE discrepancy 5.99867e-15; Stata/Python point discrepancy 2.4647e-14, SE discrepancy 2.10075e-15. All are below the 1e-9 certification threshold.

Full 999-assignment isolated Python component timing:

| estimator | point_seconds | native_seconds | jackknife_seconds | ri_seconds | replace_seconds | subtract_seconds | add_seconds | total_estimator_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastdid2spy | 0.0005 | 0.0026996 | 0.0048147 | 0.241964 | 4.58393 | 4.85122 | N/A | 9.92604 |

No 999-draw official-R or dummy-oracle placebo benchmark is claimed; each would repeatedly refit all nuisance models. Official/reference timings are compared where meaningful (point/native/JK). Twelve-worker mean complete did2s timing is 18.834465 s per replication because of CPU contention. The two-rep six-estimator wall is 33.848067 s. Eight-process fixture results and full CPS serial/parallel replay are exact.
