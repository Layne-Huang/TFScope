import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
import random

from tfscope.config import TFScopeConfig


class SyntheticTFDataset(Dataset):
    """Random dataset for testing model architecture without real data."""

    def __init__(self, config: TFScopeConfig, n_samples: int = 200,
                 max_seq_len: int = 200, seed: int = 42):
        self.config = config
        self.n_samples = n_samples
        self.max_seq_len = max_seq_len
        self.min_len = 150  # minimum protein length (ensures room for DBD)

        rng = random.Random(seed)
        self.seq_lens = [rng.randint(self.min_len, max_seq_len) for _ in range(n_samples)]
        self.dbd_starts = [rng.randint(10, sl - 70) for sl in self.seq_lens]
        self.dbd_ends = [ds + rng.randint(60, min(100, sl - ds - 10))
                         for ds, sl in zip(self.dbd_starts, self.seq_lens)]
        self.family_ids = [rng.randint(0, config.num_families - 1) for _ in range(n_samples)]
        self.motif_lengths = [rng.randint(config.min_motif_length, config.max_motif_length)
                              for _ in range(n_samples)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        L = self.seq_lens[idx]

        # Random token IDs (ESM C uses standard amino acid vocab + special tokens)
        sequence_tokens = torch.randint(4, 24, (L,))

        # DBD mask
        dbd_mask = torch.zeros(L, dtype=torch.bool)
        dbd_mask[self.dbd_starts[idx]:self.dbd_ends[idx]] = True

        # Family label
        family_id = self.family_ids[idx]

        # Target PWM: valid positions have learned probs, padding positions hold
        # uniform (0.25) so they contribute zero KL when the loss masks them out.
        motif_len = self.motif_lengths[idx]
        target_pwm = torch.full((4, self.config.max_motif_length), 0.25)
        target_pwm[:, :motif_len] = F.softmax(
            torch.randn(4, motif_len) * 0.5, dim=0
        )

        # Binary position mask: gate supervision target
        pwm_mask = torch.zeros(self.config.max_motif_length)
        pwm_mask[:motif_len] = 1.0

        return {
            'sequence_tokens': sequence_tokens,
            'dbd_mask': dbd_mask,
            'family_id': family_id,
            'target_pwm': target_pwm,
            'pwm_mask': pwm_mask,
        }


def collate_variable_length(batch):
    """Collate function for variable-length sequences."""
    max_len = max(item['sequence_tokens'].shape[0] for item in batch)
    B = len(batch)

    sequence_tokens = torch.full((B, max_len), ESM2_PAD_TOKEN, dtype=torch.long)
    dbd_mask = torch.zeros(B, max_len, dtype=torch.bool)
    has_recog = 'recog_prior' in batch[0]
    recog_prior = torch.zeros(B, max_len, dtype=torch.float32) if has_recog else None
    has_contact = 'contact_target' in batch[0]
    if has_contact:
        Lq = batch[0]['contact_target'].shape[0]
        contact_target = torch.zeros(B, Lq, max_len, dtype=torch.float32)
        contact_base_mask = torch.zeros(B, Lq, dtype=torch.float32)

    for i, item in enumerate(batch):
        L = item['sequence_tokens'].shape[0]
        sequence_tokens[i, :L] = item['sequence_tokens']
        dbd_mask[i, :L] = item['dbd_mask']
        if has_recog:
            recog_prior[i, :L] = item['recog_prior']
        if has_contact:
            contact_target[i, :, :L] = item['contact_target']
            contact_base_mask[i] = item['contact_base_mask']

    out = {
        'sequence_tokens': sequence_tokens,
        'dbd_mask': dbd_mask,
        'family_id': torch.tensor([item['family_id'] for item in batch]),
        'target_pwm': torch.stack([item['target_pwm'] for item in batch]),
        'pwm_mask': torch.stack([item['pwm_mask'] for item in batch]),
    }
    if 'retrieved_pwms' in batch[0]:
        out['retrieved_pwms']  = torch.stack([item['retrieved_pwms']  for item in batch])
        out['retrieved_masks'] = torch.stack([item['retrieved_masks'] for item in batch])
        out['retrieved_sims']  = torch.stack([item['retrieved_sims']  for item in batch])
    if has_recog:
        out['recog_prior'] = recog_prior
    if has_contact:
        out['contact_target'] = contact_target
        out['contact_base_mask'] = contact_base_mask
    return out


# Need F for softmax in __getitem__
import torch.nn.functional as F

import numpy as np
import pandas as pd


# ESM-2 amino acid token IDs (verified against esm.pretrained.esm2_t33_650M_UR50D alphabet)
# Special tokens: <cls>=0, <pad>=1, <eos>=2, <unk>=3
AA_TO_TOKEN = {
    "L": 4,  "A": 5,  "G": 6,  "V": 7,  "S": 8,  "E": 9,  "R": 10,
    "T": 11, "I": 12, "D": 13, "P": 14, "K": 15, "Q": 16, "N": 17,
    "F": 18, "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23,
    "X": 3,  "B": 3,  "Z": 3,  "U": 3,  "O": 3,   # ambiguous → <unk>=3
}
ESM2_PAD_TOKEN = 1


class GeneBalancedSampler(Sampler[int]):
    """Sample genes evenly, then choose one of each gene's motif records."""

    def __init__(
        self,
        gene_symbols,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        self.num_samples = num_samples or len(gene_symbols)
        self.seed = seed
        self.epoch = 0
        self.indices_by_gene = {}
        for index, gene_symbol in enumerate(gene_symbols):
            gene = str(gene_symbol).strip().upper()
            self.indices_by_gene.setdefault(gene, []).append(index)
        if not self.indices_by_gene:
            raise ValueError("GeneBalancedSampler requires at least one gene")
        self.genes = sorted(self.indices_by_gene)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1

        n_genes = len(self.genes)
        repeats, remainder = divmod(self.num_samples, n_genes)
        gene_positions = np.repeat(np.arange(n_genes), repeats)
        if remainder:
            extra = rng.choice(n_genes, size=remainder, replace=False)
            gene_positions = np.concatenate([gene_positions, extra])
        rng.shuffle(gene_positions)

        for gene_position in gene_positions:
            candidates = self.indices_by_gene[self.genes[int(gene_position)]]
            yield int(candidates[rng.randint(len(candidates))])

    def __len__(self):
        return self.num_samples


class TFDataset(Dataset):
    """Real TF dataset loading from processed parquet files.

    Returns dicts with the same keys as SyntheticTFDataset:
      sequence_tokens, dbd_mask, family_id, target_pwm, pwm_mask, target_length
    """

    def __init__(
        self,
        config: TFScopeConfig,
        tf_data_path: str,
        split_path: str = None,
        split: str = "train",
        max_seq_len: int = 1024,
    ):
        self.config = config
        self.max_seq_len = max_seq_len
        self.max_motif_length = config.max_motif_length
        self.split_name = split

        # Load main data table (full, unfiltered — needed as PWM donor pool for retrieval)
        df_full = pd.read_parquet(tf_data_path)

        # Build donor-pool lookup BEFORE filtering: filename → padded PWM tensor
        # This makes retrieval O(1) regardless of split membership.
        self._fn_to_padded_pwm = {}
        self._fn_to_padded_mask = {}
        for _, row in df_full.iterrows():
            pwm_bytes = row["pwm"]
            if isinstance(pwm_bytes, bytes):
                pwm_raw = np.frombuffer(pwm_bytes, dtype=np.float32).reshape(4, -1)
            else:
                pwm_raw = np.zeros((4, config.min_motif_length), dtype=np.float32)
            L = min(pwm_raw.shape[1], self.max_motif_length)
            padded = np.full((4, self.max_motif_length), 0.25, dtype=np.float32)
            padded[:, :L] = pwm_raw[:, :L]
            mask = np.zeros(self.max_motif_length, dtype=np.float32)
            mask[:L] = 1.0
            self._fn_to_padded_pwm[row["filename"]]  = padded
            self._fn_to_padded_mask[row["filename"]] = mask

        # Metadata maps for robust-RAG negative injection (v17)
        self._fn_to_family = dict(zip(df_full["filename"], df_full.get("family_name", df_full["filename"])))
        self._fn_to_gene   = dict(zip(df_full["filename"],
                                      df_full.get("gene_symbol", df_full["filename"]).astype(str).str.upper()))

        # Filter to split if provided
        if split_path is not None:
            with open(split_path) as f:
                split_data = json.load(f)
            split_ids = set(split_data[split])
            df = df_full[df_full["filename"].isin(split_ids)].reset_index(drop=True)
        else:
            df = df_full.reset_index(drop=True)

        self.df = df
        self.sequences = df["sequence"].tolist()
        self.filenames = df["filename"].tolist()

        # Pre-load all PWMs from binary blobs
        self.pwms = []
        for _, row in df.iterrows():
            pwm_bytes = row["pwm"]
            if isinstance(pwm_bytes, bytes):
                pwm = np.frombuffer(pwm_bytes, dtype=np.float32).reshape(4, -1)
            else:
                pwm = np.zeros((4, config.min_motif_length), dtype=np.float32)
            self.pwms.append(pwm)

        # Pre-extract fields for fast __getitem__
        self.family_ids = df["family_id"].tolist()
        self.dbd_starts = df["dbd_start"].tolist()
        self.dbd_ends = df["dbd_end"].tolist()
        self.seq_lengths = df["seq_length"].tolist()

        # Optional: load recognition-residue prior for the v18 contact-aware head.
        # JSON maps filename (with or without .txt) → list of 0-based sequence
        # indices that are family-canonical DNA-recognition residues.
        self.recog_prior = None
        if getattr(config, "pwm_head_v18", False):
            rp = getattr(config, "recognition_prior_path", "")
            if rp and os.path.exists(rp):
                with open(rp) as f:
                    self.recog_prior = json.load(f)

        # Optional: 2D structural contact targets (PWM-column x residue) for contact distillation.
        # JSON: filename -> {"L": seqlen, "cols": {col: [[res_idx, weight], ...]}}
        self.contact_targets = None
        if getattr(config, "contact_distill_weight", 0.0) > 0:
            cp = getattr(config, "contact_targets_path", "data/contact_maps/contact_targets.json")
            if cp and os.path.exists(cp):
                with open(cp) as f:
                    self.contact_targets = json.load(f)

        # Optional: load NN index for retrieval-augmented training
        self.nn_index = None
        if getattr(config, "use_retrieval", False):
            idx_path = config.retrieval_index_path
            if idx_path and os.path.exists(idx_path):
                with open(idx_path) as f:
                    self.nn_index = json.load(f)
            else:
                raise FileNotFoundError(
                    f"use_retrieval=True but retrieval_index_path={idx_path!r} not found"
                )

        # Robust-RAG negative pools (TRAIN split only)
        self._train_aug = (
            self.nn_index is not None and split == "train" and (
                getattr(config, "hard_negative_rate", 0.0) > 0
                or getattr(config, "all_bad_case_rate", 0.0) > 0
                or getattr(config, "neighbor_dropout", 0.0) > 0
                or getattr(config, "full_retrieval_dropout", 0.0) > 0
            )
        )
        if self._train_aug:
            donor_set = set(self._fn_to_padded_pwm.keys())
            # flattened donor PWMs for fast correlation
            self._fam_to_donors = {}
            for fn in donor_set:
                fam = self._fn_to_family.get(fn, "NA")
                self._fam_to_donors.setdefault(fam, []).append(fn)
            self._all_donors = list(donor_set)
            rng_seed = getattr(config, "seed", 42)
            self._aug_rng = np.random.RandomState(rng_seed)

    def _flat_corr(self, a, b):
        """Flattened Pearson between two (4,L) padded PWMs."""
        af, bf = a.ravel(), b.ravel()
        if af.std() < 1e-8 or bf.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(af, bf)[0, 1])

    def _sample_hard_negative(self, my_fam, my_gene, target_padded):
        """Same-family, different-gene donor with LOW PWM correlation to target."""
        cands = [fn for fn in self._fam_to_donors.get(my_fam, [])
                 if self._fn_to_gene.get(fn) != my_gene]
        if not cands:
            cands = self._all_donors
        # prefer low-correlation (hard) among a small random subset
        self._aug_rng.shuffle(cands)
        best_fn, best_c = None, 1.0
        for fn in cands[:12]:
            c = self._flat_corr(self._fn_to_padded_pwm[fn], target_padded)
            if c < best_c:
                best_c, best_fn = c, fn
            if c < 0.2:
                break
        return best_fn

    def _sample_all_bad(self, my_fam, target_padded, k):
        """k donors from DIFFERENT families (guaranteed low specificity match)."""
        other = [fn for fn in self._all_donors if self._fn_to_family.get(fn) != my_fam]
        if len(other) < k:
            other = self._all_donors
        idx = self._aug_rng.choice(len(other), size=min(k, len(other)), replace=False)
        return [other[i] for i in idx]

    def __len__(self):
        return len(self.df)

    def _tokenize(self, sequence: str) -> torch.Tensor:
        """Tokenize amino acid sequence. Falls back to simple AA mapping if ESM unavailable."""
        tokens = [AA_TO_TOKEN.get(aa, 4) for aa in sequence]
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        seq_len = min(len(sequence), self.max_seq_len)

        # Tokenize sequence
        sequence_tokens = self._tokenize(sequence)

        # DBD mask
        dbd_start = int(self.dbd_starts[idx])
        dbd_end = int(self.dbd_ends[idx])
        dbd_mask = torch.zeros(len(sequence_tokens), dtype=torch.bool)
        # Clamp to sequence length
        dbd_start = min(dbd_start, len(sequence_tokens) - 1)
        dbd_end = min(dbd_end, len(sequence_tokens))
        if dbd_end > dbd_start:
            dbd_mask[dbd_start:dbd_end] = True

        # Family label
        family_id = int(self.family_ids[idx])

        # PWM: place into fixed-size (4, max_motif_length) tensor
        pwm = self.pwms[idx]
        motif_length = min(pwm.shape[1], self.max_motif_length)

        target_pwm = torch.full((4, self.max_motif_length), 0.25, dtype=torch.float32)
        target_pwm[:, :motif_length] = torch.from_numpy(pwm[:, :motif_length].copy())

        # PWM mask
        pwm_mask = torch.zeros(self.max_motif_length, dtype=torch.float32)
        pwm_mask[:motif_length] = 1.0

        out = {
            "sequence_tokens": sequence_tokens,
            "dbd_mask": dbd_mask,
            "family_id": family_id,
            "target_pwm": target_pwm,
            "pwm_mask": pwm_mask,
        }

        # Recognition-residue prior (v18): soft per-residue target over the sequence.
        if self.recog_prior is not None:
            recog = torch.zeros(len(sequence_tokens), dtype=torch.float32)
            my_fn = self.filenames[idx]
            residues = self.recog_prior.get(my_fn) or \
                self.recog_prior.get(my_fn.replace(".txt", ""))
            if residues:
                for p in residues:
                    if 0 <= p < len(recog):
                        recog[p] = 1.0
            out["recog_prior"] = recog

        # 2D contact-distillation target: (max_motif_length, seqlen) + per-column mask
        if self.contact_targets is not None:
            my_fn = self.filenames[idx]
            entry = self.contact_targets.get(my_fn) or \
                self.contact_targets.get(my_fn.replace(".txt", ""))
            ct = torch.zeros(self.max_motif_length, len(sequence_tokens), dtype=torch.float32)
            cbm = torch.zeros(self.max_motif_length, dtype=torch.float32)
            if entry:
                for col, rows in entry["cols"].items():
                    c = int(col)
                    if not (0 <= c < self.max_motif_length):
                        continue
                    for ridx, w in rows:
                        if 0 <= ridx < ct.shape[1]:
                            ct[c, ridx] = w
                    if ct[c].sum() > 0:
                        cbm[c] = 1.0
            out["contact_target"] = ct
            out["contact_base_mask"] = cbm

        # Retrieval inputs (optional)
        if self.nn_index is not None:
            K = self.config.retrieval_k
            my_fn = self.filenames[idx]
            candidates = self.nn_index.get(my_fn, [])
            ret_pwms  = torch.full((K, 4, self.max_motif_length), 0.25, dtype=torch.float32)
            ret_masks = torch.zeros((K, self.max_motif_length), dtype=torch.float32)
            ret_sims  = torch.zeros((K,), dtype=torch.float32)
            for k_i in range(min(K, len(candidates))):
                nn_fn = candidates[k_i]["nn_filename"]
                if nn_fn == my_fn:  # extra safety: skip self
                    continue
                ret_pwms[k_i]  = torch.from_numpy(self._fn_to_padded_pwm[nn_fn])
                ret_masks[k_i] = torch.from_numpy(self._fn_to_padded_mask[nn_fn])
                ret_sims[k_i]  = float(candidates[k_i]["cos_sim"])

            # ── Robust-RAG augmentations (TRAIN split only) ──────────────────
            if getattr(self, "_train_aug", False):
                cfg = self.config
                rng = self._aug_rng
                my_fam  = self._fn_to_family.get(my_fn, "NA")
                my_gene = self._fn_to_gene.get(my_fn)
                tgt = target_pwm.numpy()

                # (4) all-bad case: replace every neighbour with a wrong-family donor
                if rng.rand() < getattr(cfg, "all_bad_case_rate", 0.0):
                    for k_i, bad in enumerate(self._sample_all_bad(my_fam, tgt, K)):
                        ret_pwms[k_i]  = torch.from_numpy(self._fn_to_padded_pwm[bad])
                        ret_masks[k_i] = torch.from_numpy(self._fn_to_padded_mask[bad])
                        # keep sims medium/high so the model can't reject by sim alone
                        ret_sims[k_i]  = float(rng.uniform(0.7, 0.95))
                else:
                    # (3) hard-negative injection: replace a few neighbours
                    if rng.rand() < getattr(cfg, "hard_negative_rate", 0.0):
                        nrep = min(getattr(cfg, "hard_negative_per_sample", 1), K)
                        for k_i in range(nrep):
                            hn = self._sample_hard_negative(my_fam, my_gene, tgt)
                            if hn is not None:
                                ret_pwms[k_i]  = torch.from_numpy(self._fn_to_padded_pwm[hn])
                                ret_masks[k_i] = torch.from_numpy(self._fn_to_padded_mask[hn])
                                ret_sims[k_i]  = float(rng.uniform(0.7, 0.95))
                    # (2) per-neighbour dropout (keep >=1 valid)
                    nd = getattr(cfg, "neighbor_dropout", 0.0)
                    if nd > 0:
                        valid = [k_i for k_i in range(K) if ret_masks[k_i].sum() > 0]
                        for k_i in list(valid):
                            if len(valid) > 1 and rng.rand() < nd:
                                ret_masks[k_i] = 0.0; ret_sims[k_i] = 0.0; valid.remove(k_i)

                # (1) full retrieval dropout: disable ALL retrieval for this sample
                if rng.rand() < getattr(cfg, "full_retrieval_dropout", 0.0):
                    ret_masks[:] = 0.0; ret_sims[:] = 0.0

            out["retrieved_pwms"]  = ret_pwms
            out["retrieved_masks"] = ret_masks
            out["retrieved_sims"]  = ret_sims

        return out
