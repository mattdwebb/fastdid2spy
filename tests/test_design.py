import unittest
import numpy as np
from fastdid2spy import Design, InferenceError, assignment_draws, FEAbsorber


def fixture(seed=1,G=10,T=8,weighted=False,covariates=False,missing=False,effect=1.,heterogeneous=False,unequal=False):
    rng=np.random.default_rng(seed); rows=[]
    a=np.array([5,5,7,7]+[0]*(G-4),float)
    for s in range(G):
        for t in range(1,T+1):
            if missing and s==G-1 and t==3:continue
            for j in range((1+s%7) if unequal else 3):
                x=rng.normal();w=np.exp(rng.normal()) if weighted else 1.
                tau=effect+(s/3-.5 if heterogeneous else 0)
                y=s/7+t/9+(0.4*x if covariates else 0)+rng.normal(scale=.25)+tau*(a[s]>0 and t>=a[s])
                rows.append((s*3+11,t,y,x,w))
    rows=np.array(rows)
    data=dict(zip(('state','year','y','x','w'),rows.T))
    design=Design(data,outcome='y',cluster='state',time='year',assignment=a,
                  covariates=['x'] if covariates else [],weights='w' if weighted else None)
    return data,a,design


class DesignTests(unittest.TestCase):
    def test_assignment_support_and_seed(self):
        observed=(5,5,0,0,0)
        exact=list(assignment_draws(observed,reps=999,seed=3))
        self.assertEqual(len(exact),9)
        self.assertEqual(len(set(map(tuple,exact))),9)
        self.assertNotIn(observed,list(map(tuple,exact)))
        a=list(assignment_draws((5,5,7,7,0,0,0,0),reps=25,seed=87))
        self.assertEqual(a,list(assignment_draws((5,5,7,7,0,0,0,0),reps=25,seed=87)))
        self.assertEqual(len(set(map(tuple,a))),25)
        self.assertTrue(all(sorted(row)==[0,0,0,0,5,5,7,7] for row in a))

    def test_unidentified_first_stage(self):
        data,a,d=fixture()
        # All states treated before the last periods: no untreated time support.
        a[:]=5
        d=Design(data,outcome='y',cluster='state',time='year',assignment=a)
        for engine in ('fast','reference'):
            with self.assertRaises(InferenceError):d.point(engine=engine)

    def test_point_and_native_grid(self):
        for seed in range(8):
            for weighted in (False,True):
                for covariates in (False,True):
                    with self.subTest(seed=seed,weighted=weighted,covariates=covariates):
                        _,_,d=fixture(seed,weighted=weighted,covariates=covariates,missing=seed%2==1,unequal=seed%3==0,heterogeneous=seed%2==0,effect=seed%2)
                        f=d.fit();r=d.fit(engine='reference')
                        np.testing.assert_allclose(f['y0'],r['y0'],atol=1e-10,rtol=0)
                        self.assertAlmostEqual(f['estimate'],r['estimate'],places=10)
                        f=d.native_cluster();r=d.native_cluster(engine='reference')
                        np.testing.assert_allclose(f['cluster_scores'],r['cluster_scores'],atol=1e-10,rtol=0)
                        self.assertAlmostEqual(f['native_cluster_se'],r['native_cluster_se'],places=10)

    def test_inference_fixed_assignments(self):
        _,a,d=fixture(44,weighted=True,covariates=True,unequal=True)
        draws=list(assignment_draws(tuple(a),reps=5,seed=89))
        for proc in ('jackknife','randomization_inference','placebo_replace','placebo_subtract'):
            kwargs={} if proc=='jackknife' else dict(assignments=draws)
            fast=getattr(d,proc)(**kwargs);ref=getattr(d,proc)(engine='reference',**kwargs)
            self.assertEqual(fast['diagnostics']['status'],'ok')
            self.assertEqual(fast['p_value'],ref['p_value']) if proc!='jackknife' else self.assertAlmostEqual(fast['p_value'],ref['p_value'],places=10)
            if proc=='jackknife':
                self.assertEqual(fast['jackknife_df'],9)
                np.testing.assert_allclose(fast['delete_estimates'],ref['delete_estimates'],atol=1e-10,rtol=0)
            elif proc=='randomization_inference':
                np.testing.assert_allclose(fast['statistics'],ref['statistics'],atol=1e-10,rtol=0)
            else:np.testing.assert_allclose(fast['procedural_p_values'],ref['procedural_p_values'],atol=1e-10,rtol=0)

    def test_scope_and_invalid(self):
        data,a,d=fixture()
        with self.assertRaises(NotImplementedError):d.placebo('add')
        data['w'][0]=-1
        with self.assertRaises(InferenceError):Design(data,outcome='y',cluster='state',time='year',assignment=a,weights='w')

    def test_singleton_cohort_surviving_att(self):
        data,a,d=fixture();a[1]=0
        d=Design(data,outcome='y',cluster='state',time='year',assignment=a)
        result=d.jackknife()
        self.assertEqual(result['jackknife_df'],9)
        self.assertEqual(result['diagnostics']['status'],'ok')
        self.assertTrue(np.isfinite(result['delete_estimates']).all())

    def test_shuffle_and_labels(self):
        data,a,d=fixture(7,weighted=True,covariates=True,missing=True)
        order=np.random.default_rng(42).permutation(len(data['y']))
        new={k:v[order] for k,v in data.items()}
        new['state']=np.array([f's{int(s):03d}' for s in new['state']])
        other=Design(new,outcome='y',cluster='state',time='year',assignment=a,covariates=['x'],weights='w')
        self.assertAlmostEqual(other.point(),d.point(),places=12)

    def test_fe(self):
        data,_,_=fixture(4,missing=True,weighted=True)
        s=data['state'];t=data['year'];v=np.column_stack([data['y'],data['x']]);w=data['w']
        fe=FEAbsorber(s,t,weights=w);r=fe.residualize(v)
        z=np.column_stack([np.eye(len(set(s)))[np.unique(s,return_inverse=True)[1]],np.eye(8)[t.astype(int)-1,1:]])
        beta=np.linalg.lstsq(z*np.sqrt(w[:,None]),v*np.sqrt(w[:,None]),rcond=None)[0]
        np.testing.assert_allclose(r,v-z@beta,atol=1e-9,rtol=0)

if __name__=='__main__':unittest.main()
