clear all
set more off
set type double
set varabbrev off
args root
cd "`root'"
adopath ++ "`root'/external/did2s_stata/ado"
file open output using "stata_results.csv", write replace
file write output "fixture,estimate,se,stat,p,df_r" _n
forvalues i=0/15 {
 local padded : display %02.0f `i'
 import delimited "fixtures/fixture_`padded'.csv", clear varnames(1) asdouble
 local cov ""
 if mod(`i',4)>=2 local cov "x"
 local wt ""
 if mod(`i',2)==1 local wt "[aw=w]"
 quietly did2s y `wt', first_stage(i.state i.year `cov') second_stage(d) treatment(d) cluster(state)
 scalar b=_b[d]
 scalar se=_se[d]
 scalar stat=scalar(b)/scalar(se)
 scalar p=2*normal(-abs(scalar(stat)))
 file write output "`i'," %24.17g (scalar(b)) "," %24.17g (scalar(se)) "," %24.17g (scalar(stat)) "," %24.17g (scalar(p)) "," %24.17g (e(df_r)) _n
}
file close output
exit, clear
