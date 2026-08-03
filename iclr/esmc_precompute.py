"""Precompute frozen ESM-C (600M) per-residue embeddings for all unique DBD
sequences, cached by sequence MD5. Runs in the `esmc` conda env (EvolutionaryScale
esm SDK). The v24_esm3c backbone (tfscope env) reads these — ESM-C cannot share the
tfscope env because its `esm` package conflicts with fair-esm (ESM-2).

Out: /data1/leihuang/TFScope_store/esmc_emb/<md5>.pt  (each = float16 (L,1152))
"""
import os, sys, hashlib, glob
import numpy as np, pandas as pd, torch

OUT = "/data1/leihuang/TFScope_store/esmc_emb"
DATA = "data/processed/tf_pwm_training_v23.parquet"


def md5(s): return hashlib.md5(s.encode()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[esmc] device={dev}", flush=True)
    m = ESMC.from_pretrained("esmc_600m").to(dev).eval()

    df = pd.read_parquet(DATA)
    seqs = sorted(set(df["sequence"].astype(str)))
    todo = [s for s in seqs if not os.path.exists(f"{OUT}/{md5(s)}.pt")]
    print(f"[esmc] {len(seqs)} unique seqs, {len(todo)} to embed", flush=True)
    with torch.no_grad():
        for i, s in enumerate(todo):
            try:
                p = ESMProtein(sequence=s[:1022])
                t = m.encode(p)
                out = m.logits(t, LogitsConfig(sequence=True, return_embeddings=True))
                emb = out.embeddings[0]                       # (L+2, 1152) incl BOS/EOS
                emb = emb[1:1 + len(s[:1022])].to(torch.float16).cpu()  # strip BOS/EOS -> (L,1152)
                torch.save(emb, f"{OUT}/{md5(s)}.pt")
            except Exception as e:
                print(f"[esmc] FAIL {md5(s)} len={len(s)}: {repr(e)[:120]}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"[esmc]   {i+1}/{len(todo)}", flush=True)
    n = len(glob.glob(f"{OUT}/*.pt"))
    print(f"[esmc] done. cached {n} embeddings in {OUT}", flush=True)


if __name__ == "__main__":
    main()
