#!/usr/bin/env python
"""Pre-compute semantic family embedding vectors for TFScope.

Each family vector = concat(text embedding, mean ESM-2 sequence embedding).

Text encoder options (--text-model):
  protrek       ProTrek_650M fine-tuned text encoder (best; run download_protrek.py first)
  pubmedbert    microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
  biobert       dmis-lab/biobert-v1.1
  scibert       allenai/scibert_scivocab_uncased
  none          skip text encoder (sequence embeddings only)

Output: /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/family_embeddings.pt
  {
    "embeddings":   Tensor (num_families, text_dim + seq_dim),
    "family_ids":   list[int],
    "family_names": list[str],
    "text_model":   str,
    "text_dim":     int,
    "seq_dim":      int,
  }

Usage:
    mamba run -n tfscope python scripts/precompute_family_embeddings.py \\
        --data       data/processed/tf_pwm.parquet \\
        --text-model protrek \\
        --out        /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/family_embeddings.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/.cache"
os.environ["HF_HOME"]            = CACHE_DIR
os.environ["TORCH_HOME"]         = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── known HuggingFace model IDs ───────────────────────────────────────────────
TEXT_MODEL_IDS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "biobert":    "dmis-lab/biobert-v1.1",
    "scibert":    "allenai/scibert_scivocab_uncased",
}

# ── family descriptions (Pfam/UniProt style) ──────────────────────────────────
FAMILY_DESCRIPTIONS = {
    "C2H2_short": (
        "C2H2-type zinc finger transcription factor with a small number of zinc finger "
        "repeats (1-3). Each finger contains a Cys2-His2 motif that coordinates a zinc "
        "ion and folds into a beta-beta-alpha structure. The alpha helix contacts the "
        "DNA major groove. Short C2H2 factors typically bind short, specific DNA motifs."
    ),
    "C2H2_medium": (
        "C2H2-type zinc finger transcription factor with an intermediate number of zinc "
        "finger repeats (4-6). Tandem C2H2 fingers cooperatively recognize extended DNA "
        "sequences. Involved in gene regulation, development, and cell differentiation. "
        "Includes classical Kruppel-associated box (KRAB) zinc finger proteins."
    ),
    "C2H2_long": (
        "C2H2-type zinc finger transcription factor with many tandem zinc finger repeats "
        "(7 or more). Each finger contributes to a long continuous DNA-binding interface "
        "recognizing extended genomic sequences. Often involves KRAB or SCAN domains for "
        "transcriptional repression and protein-protein interactions."
    ),
    "bHLH": (
        "Basic helix-loop-helix (bHLH) transcription factor. Contains a basic region for "
        "DNA binding and a helix-loop-helix domain for homo- or heterodimerization. "
        "Recognizes E-box sequences (CANNTG), particularly CACGTG (G-box) and CATCTG. "
        "Key regulators of cell differentiation, neurogenesis, myogenesis, and circadian rhythms."
    ),
    "Homeodomain": (
        "Homeodomain transcription factor containing a conserved 60-amino acid helix-turn-helix "
        "DNA-binding domain. The recognition helix contacts the DNA major groove, typically "
        "recognizing the TAAT core motif. Master regulators of body patterning, cell fate, "
        "and development. Includes HOX, PAX, LIM, and POU domain proteins."
    ),
    "bZIP": (
        "Basic leucine zipper (bZIP) transcription factor. The basic region contacts DNA "
        "and the leucine zipper mediates homo- or heterodimerization through coiled-coil "
        "interactions. Recognizes CRE (TGACGTCA) and AP-1 (TGACTCA) response elements. "
        "Regulates stress responses, metabolism, and immune function."
    ),
    "Nuclear_Receptor": (
        "Nuclear hormone receptor with a conserved C4 zinc finger DNA-binding domain. "
        "Binds hormone response elements (HREs) as homodimers or heterodimers on direct "
        "repeats, inverted repeats, or everted repeats. Ligand-activated transcription "
        "factors for steroid hormones, thyroid hormone, retinoids, and vitamin D."
    ),
    "Forkhead": (
        "Forkhead (FOX) transcription factor containing a winged-helix DNA-binding domain. "
        "The forkhead domain is a variant of the helix-turn-helix motif with butterfly-like "
        "wing structures. Recognizes the core motif GTAAACA. Involved in development, "
        "metabolism, aging, and immune function. Includes FOXA, FOXO, and FOXP subfamilies."
    ),
    "ETS": (
        "ETS domain transcription factor. The ETS domain is a winged helix-turn-helix "
        "structure that binds GGA(A/T) core sequences with flanking specificity. ETS factors "
        "are frequently involved in hematopoiesis, angiogenesis, and cancer. Includes ERG, "
        "ETV, FLI, and ELK subfamily members."
    ),
    "Other": (
        "Diverse transcription factors not belonging to the major structural families. "
        "Includes RFX, IRF, MADS-box, STAT, T-box, and other DNA-binding domain architectures. "
        "Bind a wide variety of DNA sequences and are involved in diverse biological processes "
        "including immunity, development, and signal transduction."
    ),
    # ── families recovered by the full-Pfam rebin (formerly inside 'Other') ──
    "AP2/ERF": (
        "AP2/ERF domain transcription factor. The AP2/ERF domain folds into a three-stranded "
        "antiparallel beta-sheet packed against an alpha helix; the beta-sheet inserts into the "
        "DNA major groove and recognizes GCC-box or DRE/CRT (A/GCCGAC) elements. Central to plant "
        "stress, hormone, and developmental responses; includes ERF, DREB, and AP2 subfamilies."
    ),
    "HMG/SOX": (
        "High-mobility-group (HMG) box transcription factor, including the SOX and TCF/LEF "
        "families. The HMG box is an L-shaped three-helix domain that binds the DNA minor groove "
        "at A/T-rich sites (e.g. (A/T)(A/T)CAA(A/T)G), bending the DNA sharply. Acts in sex "
        "determination, stem-cell maintenance, and Wnt signalling."
    ),
    "GATA": (
        "GATA-type zinc finger transcription factor. Each finger is a Cys4 zinc module (C-X2-C-X17-"
        "C-X2-C) whose core plus a basic tail contacts the WGATAR consensus in the DNA major and "
        "minor grooves. Master regulators of hematopoiesis, cardiac and endodermal development; "
        "includes GATA1-6."
    ),
    "p53": (
        "p53-family transcription factor (TP53, TP63, TP73). The immunoglobulin-like beta-sandwich "
        "DNA-binding domain binds, as a tetramer, two decameric RRRCWWGYYY half-sites. Central "
        "tumour suppressors governing cell-cycle arrest, apoptosis, and development."
    ),
    "IRF": (
        "Interferon-regulatory-factor (IRF) transcription factor. The IRF DNA-binding domain is a "
        "winged helix-turn-helix with a signature tryptophan cluster that recognizes the GAAA-"
        "containing ISRE/IRF-E elements. Key regulators of innate immunity and interferon response; "
        "includes IRF1-9."
    ),
    "T-box": (
        "T-box transcription factor. The large T-box domain is an immunoglobulin-like beta-barrel "
        "that binds the TCACACCT (T-box binding element) half-site, often as a dimer. Essential "
        "developmental regulators of mesoderm, heart, and limb patterning; includes TBX1-22, "
        "EOMES, and Brachyury (T)."
    ),
    "RHD/NFkB": (
        "Rel-homology-domain (RHD) transcription factor of the NF-kB/Rel/NFAT family. The "
        "immunoglobulin-like RHD binds kB sites (GGGRNNYYCC) as homo- or heterodimers. Central to "
        "inflammation, immunity, and stress signalling; includes RELA, RELB, NFKB1/2, REL, NFATC1-4."
    ),
    "Runt": (
        "Runt-domain transcription factor (RUNX1-3, CBF). The immunoglobulin-like Runt domain binds "
        "the TGYGGT (PEBP2/CBF) element and partners with CBF-beta. Master regulators of "
        "hematopoiesis, osteogenesis, and neuronal development."
    ),
    "MADS/SRF": (
        "MADS-box transcription factor (SRF, MEF2). The MADS domain is a conserved alpha-helical/"
        "beta-sheet module that binds CArG-box (CC(A/T)6GG) or MEF2 (CTA(A/T)4TAG) elements as a "
        "dimer, bending DNA. Regulators of muscle, cardiac, and serum-response gene programs."
    ),
    "E2F/DP": (
        "E2F/DP transcription factor. A winged-helix DNA-binding domain recognizes the "
        "TTTC(C/G)CGC E2F element, typically as an E2F-DP heterodimer. Central regulators of the "
        "cell cycle, DNA replication, and apoptosis; includes E2F1-8 and TFDP1-3."
    ),
    "DMRT": (
        "DMRT transcription factor with a DM (doublesex/MAB-3) DNA-binding domain. The DM domain is "
        "an intertwined zinc-binding module that contacts the DNA minor groove at a pseudopalindromic "
        "element. Conserved regulators of sexual development; includes DMRT1-3 and DMRTA/B/C."
    ),
    "STAT": (
        "Signal transducer and activator of transcription (STAT). After phosphorylation STATs dimerize "
        "and the DNA-binding domain (an immunoglobulin-like beta-barrel) binds GAS elements "
        "(TTC(N3-4)GAA). Effectors of cytokine and growth-factor JAK-STAT signalling; includes STAT1-6."
    ),
    "RFX": (
        "RFX-family transcription factor. A distinctive winged-helix DNA-binding domain binds the "
        "X-box (GTNRCC(0-3N)RGYAAC) in the DNA major and minor grooves. Regulators of MHC class II, "
        "ciliogenesis, and immune gene expression; includes RFX1-8."
    ),
    "Grainyhead/CP2": (
        "Grainyhead/CP2 (TFCP2) transcription factor. An immunoglobulin-like DNA-binding domain "
        "recognizes a pseudopalindromic element to control epithelial barrier formation, wound "
        "healing, and craniofacial development; includes GRHL1-3, TFCP2, and TFCP2L1."
    ),
    "PAX": (
        "Paired-box (PAX) transcription factor. The paired domain is a bipartite helix-turn-helix "
        "module (PAI and RED subdomains) that binds a long bipartite DNA element; many PAX factors "
        "also carry a homeodomain. Key developmental regulators of eye, kidney, and neural tissue; "
        "includes PAX1-9."
    ),
    "MYB/SANT": (
        "MYB/SANT-domain transcription factor. Tandem helix-turn-helix MYB repeats insert their "
        "third helix into the DNA major groove to recognize YAAC(G/T)G MYB elements. Regulators of "
        "proliferation, differentiation, and the circadian clock; includes MYB, MYBL1/2."
    ),
    "NF-Y/CBF": (
        "NF-Y (CBF) CCAAT-binding transcription factor. A histone-fold heterotrimer (NF-YA/NF-YB/"
        "NF-YC) clamps the CCAAT box, bending DNA and acting as a pioneer factor at promoters. "
        "Broad regulator of cell-cycle and metabolic genes."
    ),
    "GCM": (
        "Glial-cells-missing (GCM) transcription factor. The GCM domain is a beta-sheet zinc-"
        "containing module that binds the octameric ATGCGGGT motif in the DNA major groove. "
        "Regulators of gliogenesis and placental development; includes GCM1 and GCM2."
    ),
    "THAP": (
        "THAP-domain transcription factor. The THAP domain is a C2CH zinc-coordinating module with "
        "a beta-alpha-beta fold that binds specific THABS DNA elements. Functions in cell "
        "proliferation, apoptosis, and chromatin; includes THAP1 and THAP11."
    ),
    "TEA/TEAD": (
        "TEA/ATTS-domain transcription factor (TEAD1-4). The TEA domain is a three-helix bundle that "
        "binds the GGAATG MCAT element; TEAD factors are the nuclear effectors of Hippo signalling "
        "via YAP/TAZ coactivators, controlling organ size and proliferation."
    ),
    "NDT80": (
        "NDT80/PhoG-like DNA-binding transcription factor. An immunoglobulin-like beta-sandwich "
        "domain binds the MSE (gtgGACACAAAAtgg) element. Regulators of meiosis and stress response "
        "in fungi."
    ),
    "ARID/SAND": (
        "ARID/SAND-domain transcription factor. The AT-rich interaction domain (ARID) is a helix-"
        "turn-helix module that binds AT-rich DNA; SAND-domain relatives bind GGTGG/KDWK elements. "
        "Regulators of development, proliferation, and chromatin; includes ARID1-5 and SP100/AIRE-"
        "type SAND proteins."
    ),
    "MBD": (
        "Methyl-CpG-binding-domain (MBD) protein. The MBD is an alpha/beta sandwich that reads "
        "symmetrically methylated CpG dinucleotides in the DNA major groove, linking DNA "
        "methylation to transcriptional repression and chromatin. Includes MBD1-4 and MECP2."
    ),
    "HTH": (
        "Helix-turn-helix (HTH) DNA-binding protein, including the prokaryotic XRE/Cro-CI and "
        "lambda-repressor folds. A compact three-to-five-helix bundle inserts a recognition helix "
        "into the DNA major groove of a (pseudo)palindromic operator. This class also covers "
        "de-novo designed HTH binders that lie outside the natural eukaryotic TF families."
    ),
}

N_REP = 8


# ── text encoder ──────────────────────────────────────────────────────────────

class _BERTEncoder(torch.nn.Module):
    """BERT-style encoder: CLS token + optional linear projection, L2-normalised."""
    def __init__(self, bert, projection=None):
        super().__init__()
        self.bert = bert
        self.projection = projection

    def forward(self, **enc):
        cls = self.bert(**enc).last_hidden_state[:, 0, :]
        if self.projection is not None:
            cls = self.projection(cls)
        return torch.nn.functional.normalize(cls, dim=-1)


def _load_protrek(device):
    from transformers import AutoTokenizer, AutoConfig, BertModel
    protrek_dir = os.path.join(CACHE_DIR, "protrek", "ProTrek_650M_UniRef50", "text_model")
    proj_path   = os.path.join(protrek_dir, "text_projection.pt")
    meta_path   = os.path.join(protrek_dir, "protrek_meta.json")
    bin_path    = os.path.join(protrek_dir, "pytorch_model.bin")

    if not (os.path.isdir(protrek_dir) and os.path.isfile(bin_path)):
        raise FileNotFoundError(
            f"ProTrek text encoder not found at {protrek_dir}.\n"
            "Run:  mamba run -n tfscope python scripts/download_protrek.py"
        )

    print(f"Loading ProTrek text encoder from {protrek_dir}")
    tokenizer = AutoTokenizer.from_pretrained(protrek_dir)

    # AutoModel.from_pretrained calls check_torch_load_is_safe() which requires
    # torch >= 2.6 (CVE-2025-32434).  Bypass it: build model from config, then
    # load the .bin weights manually — safe because this is our own generated file.
    config = AutoConfig.from_pretrained(protrek_dir)
    bert   = BertModel(config)
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    # download_protrek.py saves keys with a leading "model." prefix; strip it.
    state_dict = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = bert.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} keys not found in saved BERT weights")

    projection = None
    if os.path.isfile(proj_path) and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        proj_dim   = meta["projection_dim"]
        proj_state = torch.load(proj_path, map_location="cpu", weights_only=False)
        has_bias   = "bias" in proj_state
        projection = torch.nn.Linear(bert.config.hidden_size, proj_dim, bias=has_bias)
        projection.weight.data = proj_state["weight"]
        if has_bias:
            projection.bias.data = proj_state["bias"]
        print(f"  + projection: {bert.config.hidden_size} → {proj_dim}")

    return tokenizer, _BERTEncoder(bert, projection).to(device).eval()


def _load_hf_bert(model_id, device):
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading text encoder: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    bert      = AutoModel.from_pretrained(model_id, cache_dir=CACHE_DIR)
    return tokenizer, _BERTEncoder(bert).to(device).eval()


def load_text_encoder(text_model: str, device):
    if text_model == "protrek":
        return _load_protrek(device)
    elif text_model in TEXT_MODEL_IDS:
        return _load_hf_bert(TEXT_MODEL_IDS[text_model], device)
    elif text_model == "none":
        return None, None
    else:
        # treat as a raw HuggingFace model ID
        return _load_hf_bert(text_model, device)


@torch.no_grad()
def encode_text(texts, tokenizer, model, device, batch_size=8):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        all_embs.append(model(**enc).cpu())
    return torch.cat(all_embs, dim=0)          # (N, H)


# ── sequence encoder (ESM-2) ──────────────────────────────────────────────────

def load_esm2(device):
    import esm as esm_lib
    model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    return model.to(device).eval(), alphabet


@torch.no_grad()
def encode_sequences(seqs, esm_model, alphabet, device, batch_size=4):
    converter = alphabet.get_batch_converter()
    all_embs  = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = [(f"seq{j}", s) for j, s in enumerate(seqs[i:i + batch_size])]
        _, _, tokens = converter(batch_seqs)
        out  = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]      # (B, L+2, 1280)
        all_embs.append(reps[:, 1:-1, :].mean(dim=1).cpu())
    return torch.cat(all_embs, dim=0)         # (N, 1280)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--data", default="data/processed/tf_pwm.parquet")
    p.add_argument("--out",
                   default="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/family_embeddings.pt")
    p.add_argument(
        "--text-model", default="protrek",
        choices=["protrek", "pubmedbert", "biobert", "scibert", "none"],
        help=(
            "Text encoder for family descriptions. "
            "'protrek' uses the ProTrek_650M fine-tuned encoder (best). "
            "'none' skips text and uses sequence embeddings only."
        ),
    )
    p.add_argument("--n-rep", type=int, default=N_REP,
                   help="Representative sequences per family for ESM-2 mean")
    p.add_argument("--cpu", action="store_true", help="Force CPU (slow but safe)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Text model: {args.text_model}")

    df = pd.read_parquet(args.data)
    family_map = (df.drop_duplicates("family_id")
                    .sort_values("family_id")
                    [["family_id", "family_name"]]
                    .reset_index(drop=True))
    family_ids   = family_map["family_id"].tolist()
    family_names = family_map["family_name"].tolist()
    print(f"Families: {list(zip(family_ids, family_names))}")

    parts      = []
    text_dim   = 0
    seq_dim    = 0

    # ── 1. Text embeddings ────────────────────────────────────────────────────
    if args.text_model != "none":
        print(f"\n── Text encoder ({args.text_model}) ──")
        tok, txt_model = load_text_encoder(args.text_model, device)
        descriptions   = [FAMILY_DESCRIPTIONS.get(n, f"Transcription factor family: {n}")
                          for n in family_names]
        text_embs = encode_text(descriptions, tok, txt_model, device)
        text_dim  = text_embs.shape[1]
        print(f"Text embeddings: {text_embs.shape}")
        parts.append(text_embs)
        del txt_model
    else:
        print("\n── Skipping text encoder (--text-model none) ──")

    # ── 2. Sequence embeddings ────────────────────────────────────────────────
    print("\n── ESM-2 sequence encoder ──")
    esm_model, alphabet = load_esm2(device)
    seq_embs_list = []
    for fid, fname in zip(family_ids, family_names):
        fdf = df[df["family_id"] == fid].drop_duplicates("uniprot_id").sort_values("seq_length")
        n   = min(args.n_rep, len(fdf))
        idx = np.round(np.linspace(0, len(fdf) - 1, n)).astype(int)
        rep_seqs = [s[:1024] for s in fdf.iloc[idx]["sequence"].tolist()]
        embs = encode_sequences(rep_seqs, esm_model, alphabet, device)
        seq_embs_list.append(embs.mean(0))
        print(f"  {fname}: {len(rep_seqs)} seqs → {embs.shape}")
    seq_embs = torch.stack(seq_embs_list, dim=0)
    seq_dim  = seq_embs.shape[1]
    print(f"Sequence embeddings: {seq_embs.shape}")
    parts.append(seq_embs)
    del esm_model

    # ── 3. Concatenate & save ─────────────────────────────────────────────────
    combined = torch.cat(parts, dim=1)
    print(f"\nCombined embeddings: {combined.shape}  "
          f"(text={text_dim}, seq={seq_dim})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "embeddings":   combined,
        "family_ids":   family_ids,
        "family_names": family_names,
        "text_model":   args.text_model,
        "text_dim":     text_dim,
        "seq_dim":      seq_dim,
    }, args.out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
