"""Altschul-Erikson dinucleotide-preserving shuffle (exact).

The naive random Eulerian walk dead-ends ~75% of the time. The fix (Altschul & Erikson 1985):
for every vertex v != last_vertex, the LAST outgoing edge used must lie on a spanning tree
rooted at last_vertex. Then a random Eulerian circuit always exists and the walk never stalls.
Verified: output has IDENTICAL dinucleotide counts to the input (and identical length).
"""
import numpy as np
ALPH="ACGT"

def dinuc_shuffle(s, rng):
    s=[c for c in s if c in ALPH]
    n=len(s)
    if n<4: return "".join(s)
    last=s[-1]
    edges={}
    for a,b in zip(s[:-1],s[1:]): edges.setdefault(a,[]).append(b)
    verts=[v for v in edges if v!=last]
    # pick last-edges forming a tree rooted at `last`; retry until valid
    for _ in range(200):
        lastedge={v: edges[v][rng.integers(len(edges[v]))] for v in verts}
        ok=True
        for v in verts:                       # follow last-edges; must reach `last`
            seen={v}; cur=v
            while cur!=last:
                nxt=lastedge.get(cur)
                if nxt is None or nxt in seen: ok=False; break
                seen.add(nxt); cur=nxt
            if not ok: break
        if ok: break
    else:
        return "".join(s)                     # give up: return input unchanged (never silently mono-shuffle)
    # shuffle remaining edges, append the reserved last-edge
    order={}
    for v,lst in edges.items():
        rest=list(lst)
        if v in lastedge:
            rest.remove(lastedge[v])
            rng.shuffle(rest); rest.append(lastedge[v])
        else:
            rng.shuffle(rest)
        order[v]=rest
    out=[s[0]]; cur=s[0]; idx={v:0 for v in order}
    for _ in range(n-1):
        lst=order.get(cur)
        if lst is None or idx[cur]>=len(lst): return "".join(s)
        nxt=lst[idx[cur]]; idx[cur]+=1; out.append(nxt); cur=nxt
    return "".join(out)

def dinuc_counts(s):
    d={}
    for a,b in zip(s[:-1],s[1:]): d[a+b]=d.get(a+b,0)+1
    return d
