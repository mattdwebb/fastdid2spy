# Corrected native clustered covariance

Let A=X'WUX, B=D'WD, H=D'WX, e1=Y-X beta, and e2=e1-D theta. Cluster s contributes

`q_s = sum_{i in s} [w_i D_i e2_i - H A^{-1} X_i w_i U_i e1_i] / B`.

The variance is `sum_s q_s^2`. The second term accounts for estimation of the untreated first stage. Neither naive second-stage clustering nor the jackknife substitutes for this score. The production implementation obtains the same score after eliminating state nuisance parameters; the reference forms the full matrices. Cluster scores are returned for direct comparison. Weights enter both moment equations and cluster sums as positive WLS observation weights; there is no frequency-weight expansion.

No G/(G-1), N/(N-K), or other finite-sample variance multiplier is applied: this follows the pinned official R and Stata source. The paper's asymptotic statistic and Stata post-estimation convention are normal. `native_cluster()` defaults to two-sided normal p-values and df=None. Official R did2s supplies a custom covariance to fixest summary; in the pinned R/fixest 0.14.2 configuration its scalar summary uses t(N-1). `reference_distribution="r_t"` reproduces that p-value, with the same estimate, SE and statistic. This is a software presentation difference, not a covariance disagreement. It must not be silently forced to t(G-1); only this project's jackknife is required to use t(G-1).

External matching uses first_stage state/time FE plus identical optional covariates; second_stage `~0+d`; identical positive weights, state clustering and sample. Official R's default FE convergence tolerance initially gave about 1.08e-7 point discrepancy. Setting fixest's permitted tolerance to 3e-12 yielded approximately 1.37e-13 point and 1.39e-14 SE discrepancies without changing formulas or data. Stata uses built-in regress/Mata in its official did2s implementation. All fixtures use double precision. See certification result files for exact maxima.
