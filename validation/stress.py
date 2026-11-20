import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,pathlib,json,time,concurrent.futures
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent;P=ROOT.parent/'fastdid2spy'
sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'tests'))
from test_design import fixture
from fastdid2spy import Design,assignment_draws,InferenceError

def make_design(name):
    options=dict(seed=823,weighted='weight' in name,covariates='cov' in name,
                 missing='missing' in name,unequal='unequal' in name,G=50 if 'fifty' in name else 10,
                 effect=0. if 'null' in name else 1.,heterogeneous='hetero' in name)
    data,a,d=fixture(**options)
    if name=='one_cohort':a[a==7]=5
    if name=='large_cohorts':a[:8]=[5]*4+[7]*4
    if name=='singleton':a[1]=0
    if name=='weak_null':
        D=(a[d._rowstate]>0)&(d.times[d._rowtime]>=a[d._rowstate]);tau=np.where(d._rowstate%2,1.,-1.)
        tau=tau-np.average(tau[D],weights=d.w[D]);data['y']+=tau*D
    if name=='near_cov':
        data['x2']=data['x']+np.random.default_rng(15).normal(scale=1e-5,size=len(data['x']))
    d=Design(data,outcome='y',cluster='state',time='year',assignment=a,
             covariates=['x','x2'] if name=='near_cov' else ['x'] if options['covariates'] else [],
             weights='w' if options['weighted'] else None)
    return d

def process_job(seed):
    _,a,d=fixture(seed=seed,weighted=True,covariates=True)
    assignments=list(assignment_draws(tuple(a),reps=4,seed=seed+100))
    return [d.point(),d.native_cluster()['native_cluster_se'],d.jackknife()['p_value'],
            d.randomization_inference(assignments=assignments)['p_value'],
            d.placebo_replace(assignments=assignments)['p_value'],d.placebo_subtract(assignments=assignments)['p_value']]

def main():
    rows=[]
    names=['balanced','unequal','missing','fifty','large_cohorts','one_cohort','null','hetero','weak_null',
           'cov','weight','weight_cov_unequal','near_cov','singleton']
    for name in names:
        d=make_design(name);start=time.perf_counter();f=d.fit();r=d.fit(engine='reference')
        point=abs(f['estimate']-r['estimate']);prediction=float(np.max(abs(f['y0']-r['y0'])))
        assert point<1e-9 and prediction<1e-8,(name,point,prediction)
        native=d.native_cluster();nr=d.native_cluster(engine='reference');nerr=abs(native['native_cluster_se']-nr['native_cluster_se'])
        assert nerr<1e-9,(name,nerr)
        jk=d.jackknife();jr=d.jackknife(engine='reference')
        if name=='singleton':
            assert jk['diagnostics']['status']==jr['diagnostics']['status']=='singular_jackknife_replication'
            rows.append(dict(design=name,status='expected_undefined_jackknife',point_error=point,native_se_error=nerr));continue
        assert jk['diagnostics']['status']==jr['diagnostics']['status']=='ok'
        jerr=float(np.max(abs(jk['delete_estimates']-jr['delete_estimates'])))
        assert jerr<1e-9 and jk['jackknife_df']==len(d.states)-1,(name,jerr)
        draws=list(assignment_draws(tuple(d.assignment),reps=3,seed=541))
        errors=[]
        for procedure in ('randomization_inference','placebo_replace','placebo_subtract'):
            f=getattr(d,procedure)(assignments=draws);r=getattr(d,procedure)(assignments=draws,engine='reference')
            assert f['diagnostics']['status']==r['diagnostics']['status']=='ok',(name,procedure)
            assert f['hits']==r['hits'] and f['p_value']==r['p_value'],(name,procedure)
            field='statistics' if procedure=='randomization_inference' else 'procedural_p_values'
            errors.append(float(np.max(abs(f[field]-r[field]))))
        rows.append(dict(design=name,status='pass',point_error=point,native_se_error=nerr,jackknife_max_error=jerr,
                         assignment_max_error=max(errors),seconds=time.perf_counter()-start))
    pd.DataFrame(rows).to_csv(P/'docs/stress_test_results.csv',index=False)
    seeds=list(range(410,418));serial=list(map(process_job,seeds))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:parallel=list(pool.map(process_job,seeds))
    assert serial==parallel
    (P/'docs/process_isolation.json').write_text(json.dumps(dict(workers=8,seeds=seeds,exact_match=True,serial=serial),indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print('8-process exact match')

if __name__=='__main__':main()
