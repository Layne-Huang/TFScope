"""MyoD1 L112R specificity-switch (WT muscle E-box CACCTG -> mutant canonical CACGTG)
for the two new training versions (rag_contact, perprotein_text) vs combined reference.
Reports predicted consensus + E-box central dinucleotide for WT and L112R.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.path.insert(0, "src")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

WT  = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
MUT = "RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
B = np.array(list("ACGT")); dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPTS = {
 "combined (learned-10, ref)": ("/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt", 3, None),
 "rag_contact":                 ("/data1/leihuang/project/TFScope/checkpoints/v19_combined_rag_contact/rag_seed42/ckpt_best.pt", 3, None),
 "perprotein_text":             ("/data1/leihuang/project/TFScope/checkpoints/v19_combined_perprotein_text/rag_seed42/ckpt_best.pt", None,
                                 "Transcription factor MYOD1 from Homo sapiens, bHLH family."),
}

def load(ckpt):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    cfg.use_retrieval = False
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"], strict=False)
    return m

_bert = None
def text_vec(text):
    global _bert
    from transformers import AutoTokenizer, AutoModel
    if _bert is None:
        mid = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"; CD = "/data1/leihuang/.cache"
        tk = AutoTokenizer.from_pretrained(mid, cache_dir=CD, local_files_only=True)
        bt = AutoModel.from_pretrained(mid, cache_dir=CD, local_files_only=True).to(dev).eval()
        _bert = (tk, bt)
    tk, bt = _bert
    with torch.no_grad():
        e = {k: v.to(dev) for k, v in tk(text, return_tensors="pt", truncation=True, max_length=128).items()}
        return F.normalize(bt(**e).last_hidden_state[:, 0, :], dim=-1)[0]

def predict(m, seq, fid, text):
    if text is not None:                       # per-protein: append text vec, use offset id
        fe = m.moe.family_embed
        if fe.vectors.shape[0] == _ntrain[0]:  # append once
            fe.vectors = torch.cat([fe.vectors, text_vec(text)[None].to(fe.vectors.dtype).to(dev)], 0)
        fid = fe.vectors.shape[0] - 1
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([fid], dtype=torch.long, device=dev)
    with torch.no_grad():
        gl, pl, _ = m(tok, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
        gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    L = max(4, int((gate > 0.5).sum())); return pwm[:, :L]

def ebox(core):
    con = "".join(B[core.argmax(0)])
    for i in range(len(con) - 5):
        w = con[i:i+6]
        if w[0]=="C" and w[1]=="A" and w[4]=="T" and w[5]=="G": return con, w[2:4]
    return con, None

print(f"{'model':<28} {'WT consensus':<16} {'WT Ebox':<9} {'MUT consensus':<16} {'MUT Ebox':<9} switch")
print("-"*95)
for name,(ckpt,fid,text) in CKPTS.items():
    m = load(ckpt); _ntrain = [m.moe.family_embed.vectors.shape[0]] if hasattr(m.moe.family_embed,"vectors") else [None]
    cw = predict(m, WT, fid, text); cm = predict(m, MUT, fid, text)
    conw,ebw = ebox(cw); conm,ebm = ebox(cm)
    sw = f"CA[{ebw}]TG" if ebw else "no Ebox"; sm = f"CA[{ebm}]TG" if ebm else "no Ebox"
    got = "G recovered" if ebm in ("CG","GC") and "G" in (ebm or "") and ebm!=ebw else ("→"+str(ebm))
    print(f"{name:<28} {conw[:15]:<16} {sw:<9} {conm[:15]:<16} {sm:<9} {got}")
print("\nbiology: WT -> CACCTG/CAGCTG (CC/GC) ; L112R -> CACGTG (CG). Recovering G at the switch = correct.")
