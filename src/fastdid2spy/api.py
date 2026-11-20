"""Static did2s: untreated WLS then adjusted outcome on D, without intercept.

Numerical code is NumPy-only. Data may be a mapping of arrays or a DataFrame.
The bounded dummy oracle and the state-Schur production path are independent.
"""
import math
import numpy as np
from ._inference import InferenceMixin, InferenceError
from ._probability import student_t
from ._fe import FEAbsorber, ConvergenceError


class Design(InferenceMixin):
    kind = "did2s"

    def __init__(self, data, *, outcome, cluster, time, assignment, covariates=(), weights=None):
        self.outcome, self.cluster, self.time = outcome, cluster, time
        self.covariates = list(covariates)
        self.weight_name = weights
        self.y = np.asarray(data[outcome], float).copy()
        labels = np.asarray(data[cluster]); years = np.asarray(data[time], float)
        self.x = np.column_stack([np.asarray(data[v], float) for v in covariates]) if covariates else np.empty((len(self.y), 0))
        self.w = np.ones(len(self.y)) if weights is None else np.asarray(data[weights], float).copy()
        if len(self.y) == 0 or any(len(v) != len(self.y) for v in (labels, years, self.w)):
            raise InferenceError("invalid_sample", "empty or inconsistent input lengths")
        if not all(np.isfinite(v).all() for v in (self.y, years, self.x, self.w)) or np.any(self.w <= 0):
            raise InferenceError("invalid_sample", "finite data and strictly positive WLS weights required")
        if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) for v in labels):
            raise InferenceError("invalid_sample", "missing state label")
        self.states = sorted(set(labels.tolist())); self.times = np.unique(years)
        if len(self.times)<2: raise InferenceError("invalid_sample", "at least two time periods required")
        mapping={s:i for i,s in enumerate(self.states)}
        self._rowstate = np.array([mapping[v] for v in labels])
        self._rowtime = np.searchsorted(self.times, years)
        self.assignment = self._assignment(assignment)
        self._compile()

    def _compile(self):
        G, T = len(self.states), len(self.times)
        self._cell = self._rowstate*T + self._rowtime
        p = T-1+self.x.shape[1]; C = G*T
        # Time indicators have T-1 columns, never N by number-of-state-levels.
        q = np.column_stack([np.eye(T)[self._rowtime, 1:], self.x])
        def sums(v):
            return np.bincount(self._cell, weights=v, minlength=C).reshape(G,T)
        self.counts = sums(self.w); self.ysum = sums(self.w*self.y)
        self.zsum = np.stack([sums(self.w*q[:,j]) for j in range(p)], axis=2)
        self.zy = np.stack([sums(self.w*q[:,j]*self.y) for j in range(p)], axis=2)
        self.zz = np.empty((G,T,p,p))
        for j in range(p):
            for k in range(p): self.zz[:,:,j,k] = sums(self.w*q[:,j]*q[:,k])
        self._cache = None

    def _blocks(self, a):
        key = tuple(a)
        if self._cache is not None and self._cache[0] == key: return self._cache[1]
        treated = (a[:,None]>0)&(self.times[None,:]>=a[:,None]); u = ~treated
        n = (self.counts*u).sum(1); sy = (self.ysum*u).sum(1)
        if np.any(n <= 0): raise InferenceError("rank_deficient", "state has no untreated observations")
        sz = np.einsum('gtp,gt->gp',self.zsum,u)
        qy = np.einsum('gtp,gt->gp',self.zy,u)
        qq = np.einsum('gtpq,gt->gpq',self.zz,u)
        S = qq-sz[:,:,None]*sz[:,None,:]/n[:,None,None]
        b = qy-sz*sy[:,None]/n[:,None]
        tn=(self.counts*treated).sum(1); ty=(self.ysum*treated).sum(1)
        tz=np.einsum('gtp,gt->gp',self.zsum,treated)
        result=(n,sy,sz,S,b,S.sum(0),b.sum(0),tn,ty,tz)
        self._cache=(key,result)
        return result

    def _fast(self,a,omit=None,details=False):
        n,sy,sz,S,b,lhs,rhs,tn,ty,tz=self._blocks(a)
        keep=np.ones(len(n),bool)
        if omit is not None:
            keep[omit]=False; lhs=lhs-S[omit]; rhs=rhs-b[omit]
        mass=tn[keep].sum()
        if mass<=0: raise InferenceError("undefined_estimand", "no treated observations")
        scale=np.sqrt(np.maximum(np.diag(lhs),0))
        if np.any(scale<=np.sqrt(np.finfo(float).eps)*max(1.,scale.max())):
            return self._absorbed(a,omit,details)
        scaled=lhs/scale[:,None]/scale[None,:]; eig=np.linalg.eigvalsh(scaled)
        if eig[0]<=1e-9*eig[-1]: return self._absorbed(a,omit,details)
        gamma=np.linalg.solve(scaled,rhs/scale)/scale
        alpha=(sy-sz@gamma)/n
        theta=float((ty[keep]-tn[keep]*alpha[keep]-tz[keep]@gamma).sum()/mass)
        if not details:return theta
        y0=alpha[self._rowstate]+np.r_[0.,gamma[:len(self.times)-1]][self._rowtime]+self.x@gamma[len(self.times)-1:]
        return dict(estimate=theta,alpha=alpha,gamma=gamma,y0=y0,solver="state_schur",lhs=lhs,keep=keep)

    def _reference(self,a,omit=None,details=False):
        keep=np.ones(len(self.y),bool) if omit is None else self._rowstate!=omit
        active=np.unique(self._rowstate[keep]); G=len(self.states);T=len(self.times)
        if len(self.y)*(len(active)+T-1+self.x.shape[1])>5_000_000:
            raise InferenceError("oracle_size_limit", "explicit dummies are a small-sample reference only")
        z=np.column_stack([np.eye(G)[self._rowstate][:,active],np.eye(T)[self._rowtime,1:],self.x])
        d=(a[self._rowstate]>0)&(self.times[self._rowtime]>=a[self._rowstate]); u=keep&~d
        if not np.any(keep&d):raise InferenceError("undefined_estimand")
        beta,_,rank,_=np.linalg.lstsq(z[u]*np.sqrt(self.w[u,None]),self.y[u]*np.sqrt(self.w[u]),rcond=None)
        if rank<z.shape[1]:raise InferenceError("rank_deficient")
        y0=z@beta; theta=float(np.sum(self.w[keep]*d[keep]*(self.y-y0)[keep])/np.sum(self.w[keep]*d[keep]))
        if not details:return theta
        return dict(estimate=theta,y0=y0,z=z,untreated=u,treated=d,keep=keep,coefficients=beta,solver="explicit_dummy_lstsq")

    def _absorbed(self,a,omit=None,details=False):
        keep=np.ones(len(self.y),bool) if omit is None else self._rowstate!=omit
        d=(a[self._rowstate]>0)&(self.times[self._rowtime]>=a[self._rowstate]); u=keep&~d
        try:
            fe=FEAbsorber(self._rowstate[u],self._rowtime[u],weights=self.w[u])
            if fe.components != 1: raise InferenceError("rank_deficient", "disconnected untreated FE design is outside this estimator's full-rank specification")
            values=np.column_stack([self.y,self.x])
            residual,diag,effects=fe.residualize(values[u],tol=1e-13,return_effects=True)
            beta,_,rank,_=np.linalg.lstsq(residual[:,1:]*np.sqrt(self.w[u,None]),residual[:,0]*np.sqrt(self.w[u]),rcond=None)
            if rank<self.x.shape[1]:raise InferenceError("rank_deficient")
            pred=fe.predict_effects(effects,self._rowstate[keep],self._rowtime[keep])
            y0=np.full(len(self.y),np.nan);y0[keep]=pred[:,0]+(self.x[keep]-pred[:,1:])@beta
            target=keep&d; theta=float(np.average((self.y-y0)[target],weights=self.w[target]))
        except ConvergenceError as e:
            raise InferenceError("fe_nonconvergence",str(e)) from e
        except (ValueError,KeyError,np.linalg.LinAlgError) as e:
            raise InferenceError("rank_deficient",str(e)) from e
        if not details:return theta
        return dict(estimate=theta,y0=y0,solver="FEABSORB-1",fe_diagnostics=diag)

    def _estimate(self,a,engine,omit,procedure,observed):
        if procedure!='replace':raise NotImplementedError("Add is intentionally unsupported for did2s")
        return self._reference(a,omit) if engine=='reference' else self._fast(a,omit)

    def fit(self,*,engine="fast"):
        return self._reference(self.assignment,details=True) if engine=="reference" else self._fast(self.assignment,details=True)

    def _subtract(self,value):
        import copy
        out=copy.copy(self)
        d=(self.assignment[self._rowstate]>0)&(self.times[self._rowtime]>=self.assignment[self._rowstate])
        out.y=self.y-value*d;out._compile()
        return out

    def native_cluster(self,*,null=0.,engine="fast",reference_distribution="normal"):
        """Unscaled cluster GMM sandwich; normal inference (paper/Stata).

        R fixest's custom covariance summary can be reproduced with 'r_t'.
        This changes only the reference distribution, never the covariance.
        """
        a=self.assignment
        if engine=="reference":
            fit=self._reference(a,details=True);z=fit['z'];u=fit['untreated'];d=fit['treated']
            mass=float(self.w@d);e=self.y-fit['y0'];theta=fit['estimate']
            A=z[u].T@(self.w[u,None]*z[u]); h=z.T@(self.w*d)
            correction=np.linalg.solve(A,h)
            scores=self.w*(d*(e-theta*d)-u*e*(z@correction))/mass
            cluster_scores=np.bincount(self._rowstate,weights=scores,minlength=len(self.states))
        elif engine=="fast":
            fit=self._fast(a,details=True)
            if fit['solver']!='state_schur':
                # Numerical stability fallback for native covariance is bounded explicitly.
                return self.native_cluster(null=null,engine="reference",reference_distribution=reference_distribution)
            n,sy,sz,S,b,lhs,rhs,tn,ty,tz=self._blocks(a)
            theta=fit['estimate'];gamma=fit['gamma'];alpha=fit['alpha'];mass=tn.sum()
            h=(tz-tn[:,None]*sz/n[:,None]).sum(0)
            delta=np.linalg.solve(lhs,h)
            first_score=b-np.einsum('gpq,q->gp',S,gamma)
            second_score=ty-tn*alpha-tz@gamma-tn*theta
            cluster_scores=(second_score-first_score@delta)/mass
        else:raise ValueError("engine must be fast or reference")
        se=float(np.linalg.norm(cluster_scores));stat=(theta-null)/se if se>0 else float('nan')
        if reference_distribution=="normal":df=None;p=math.erfc(abs(stat)/math.sqrt(2))
        elif reference_distribution=="r_t":df=len(self.y)-1;p=float(2*student_t.sf(abs(stat),df))
        else:raise ValueError("reference_distribution must be normal or r_t")
        return dict(estimate=theta,native_cluster_method="did2s_corrected_cluster_GMM",
                    native_cluster_se=se,native_cluster_stat=stat,native_cluster_df=df,native_cluster_p=p,
                    native_cluster_distribution=reference_distribution,cluster_scores=cluster_scores,
                    finite_sample_multiplier=1.,status="ok" if se>0 else "zero_standard_error")
