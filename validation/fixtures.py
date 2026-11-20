import sys, pathlib,json
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
PACKAGE=ROOT.parent/'fastdid2spy'
sys.path.insert(0,str(PACKAGE/'src'));sys.path.insert(0,str(PACKAGE/'tests'))
from test_design import fixture
rows=[]
for i in range(16):
    data,a,d=fixture(seed=100+i,weighted=i%2==1,covariates=i%4>=2,missing=i%3==0,unequal=i%3==1,heterogeneous=i%4==0,effect=i%2)
    frame=pd.DataFrame(data); frame['g']=a[d._rowstate];frame['D']=((frame.g>0)&(frame.year>=frame.g)).astype(int)
    folder=ROOT/'fixtures';folder.mkdir(exist_ok=True)
    frame.to_csv(folder/f'fixture_{i:02}.csv',index=False,float_format='%.17g')
    fast=d.native_cluster();ref=d.native_cluster(engine='reference');fit=d.fit();rf=d.fit(engine='reference')
    rows.append(dict(fixture=i,weighted=i%2==1,covariates=i%4>=2,estimate=fast['estimate'],se=fast['native_cluster_se'],stat=fast['native_cluster_stat'],p_normal=fast['native_cluster_p'],p_r=d.native_cluster(reference_distribution='r_t')['native_cluster_p'],n=len(frame),max_y0_error=float(np.max(abs(fit['y0']-rf['y0']))),fast_reference_estimate_error=abs(fast['estimate']-ref['estimate']),fast_reference_se_error=abs(fast['native_cluster_se']-ref['native_cluster_se'])))
pd.DataFrame(rows).to_csv(ROOT/'python_results.csv',index=False,float_format='%.17g')
