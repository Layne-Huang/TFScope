"""Eulerian (Altschul-Erikson 1985) dinucleotide-preserving shuffle: a surrogate with the
EXACT same dinucleotide composition as the input, but randomized order. Falls back to
identity for very short / single-symbol sequences. N runs are kept verbatim per-segment.
"""
import numpy as np
from collections import defaultdict


def _euler_shuffle(s, rng):
    if len(s) < 3:
        return s
    last = s[-1]
    # outgoing edges per vertex (as dinucleotides s[i] -> s[i+1])
    edges = defaultdict(list)
    for i in range(len(s) - 1):
        edges[s[i]].append(s[i + 1])
    verts = list(edges.keys())
    for _ in range(20):                          # retry until last-edges form a tree to `last`
        last_edge = {}
        rest = {v: list(e) for v, e in edges.items()}
        ok = True
        for v in verts:
            if v == last:
                continue
            choices = rest[v]
            j = rng.integers(len(choices))
            last_edge[v] = choices[j]
            rest[v] = choices[:j] + choices[j + 1:]
        # check the last_edge graph is a tree rooted at `last` (no cycles among non-last verts)
        for v in verts:
            if v == last:
                continue
            seen = set(); cur = v
            while cur != last:
                if cur in seen or cur not in last_edge:
                    ok = False; break
                seen.add(cur); cur = last_edge[cur]
            if not ok:
                break
        if not ok:
            continue
        # assemble: shuffle the remaining edges, append the tree last-edge at the end of each list
        for v in verts:
            rng.shuffle(rest[v])
            if v != last:
                rest[v].append(last_edge[v])
        out = [s[0]]; cur = s[0]
        idx = {v: 0 for v in verts}
        for _ in range(len(s) - 1):
            nxt = rest[cur][idx[cur]]; idx[cur] += 1
            out.append(nxt); cur = nxt
        return "".join(out)
    return s                                      # give up → return original (rare)


def dinucleotide_shuffle(seq, rng):
    """Shuffle within maximal ACGT runs (N-runs preserved in place)."""
    seq = seq.upper()
    out = []; i = 0; n = len(seq)
    while i < n:
        if seq[i] in "ACGT":
            j = i
            while j < n and seq[j] in "ACGT":
                j += 1
            out.append(_euler_shuffle(seq[i:j], rng)); i = j
        else:
            out.append(seq[i]); i += 1
    return "".join(out)


if __name__ == "__main__":   # self-test: dinucleotide counts must be preserved
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(200):
        L = rng.integers(50, 600)
        s = "".join(rng.choice(list("ACGT"), L))
        sh = dinucleotide_shuffle(s, rng)
        def di(x):
            d = defaultdict(int)
            for k in range(len(x) - 1): d[x[k:k + 2]] += 1
            return dict(d)
        if di(s) != di(sh) or s[0] != sh[0] or s[-1] != sh[-1] or len(s) != len(sh):
            bad += 1
    print(f"self-test: {bad}/200 sequences with mismatched dinucleotide composition (want 0)")
