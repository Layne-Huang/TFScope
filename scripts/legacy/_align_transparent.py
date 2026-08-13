"""Transparent ungapped alignment: slide pred over ref, score ONLY overlapping columns.
No uniform padding is ever scored (align_pwm pads with uniform columns and includes them,
which inflates/deflates r). Returns best forward and best RC alignment separately."""
import numpy as np
def _rc(p): return p[::-1,::-1].copy()
def _colr(a,b):
    if np.std(a)<1e-8 or np.std(b)<1e-8: return np.nan
    return float(np.corrcoef(a,b)[0,1])
def best_align(pred, ref, min_overlap=5):
    Lp,Lr=pred.shape[1],ref.shape[1]
    out={}
    for tag,P in [("fwd",pred),("rc",_rc(pred))]:
        best=None
        for off in range(-(Lp-min_overlap), Lr-min_overlap+1):
            i0=max(0,-off); i1=min(Lp,Lr-off)
            if i1-i0<min_overlap: continue
            rs=[_colr(P[:,i],ref[:,i+off]) for i in range(i0,i1)]
            rs=[x for x in rs if not np.isnan(x)]
            if not rs: continue
            r=float(np.mean(rs))
            if best is None or r>best[0]:
                pc="".join("ACGT"[k] for k in P[:,i0:i1].argmax(0))
                rc_="".join("ACGT"[k] for k in ref[:,i0+off:i1+off].argmax(0))
                best=(r,off,i1-i0,pc,rc_)
        out[tag]=best
    return out
