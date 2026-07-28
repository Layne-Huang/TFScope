"""Recognition-Energy Decoder (Phase 3).

Standalone frozen-ESM residue reps -> PWM via an explicit, low-rank residue->base
recognition energy:

    z[j,b] = r[fam][b,j]                                  # low-capacity family prior
           + lam_d * sum_i C[j,i] * phi(a_i, h_i)[b]      # direct readout
           + lam_s * sum_i C[j,i] sum_{k in N(i)} A[i,k] * psi(a_i,a_k,h_i,h_k)[b]

    PWM = softmax_b z

Design contract:
- standalone sequence input (h = frozen ESM residue embs, a = AA one-hot);
- family prior r is LOW capacity (n_fam x 4 x n_pos), so it cannot memorise per-gene
  motifs and cannot swamp the residue->base term;
- the mutation-sensitive pathway (phi, psi, C) NEVER reads the family id/embedding, so a
  point mutation's effect flows purely through the residue reps;
- C is a SOFT contact (attention, optional additive oracle bias) -- no hard mask;
- direct (phi) and second-shell (psi) are independently ablatable;
- no gene/mutation-specific rules;
- per-residue->base energy contributions are returned for interpretation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecognitionEnergyDecoder(nn.Module):
    def __init__(self, esm_dim=1280, d=96, n_pos=20, n_fam=1, aa_dim=20,
                 nbhd=2, use_second_shell=True):
        super().__init__()
        self.n_pos, self.aa_dim, self.nbhd = n_pos, aa_dim, nbhd
        self.use_second_shell = use_second_shell
        # low-capacity family prior (bHLH: n_fam=1 -> 4*n_pos params)
        self.r = nn.Parameter(torch.zeros(n_fam, 4, n_pos))
        # residue projection (shared)
        self.proj = nn.Sequential(nn.Linear(esm_dim, d), nn.LayerNorm(d), nn.GELU())
        # soft contact C[j,i]: position query e_j attends to key(h_i)
        self.pos_q = nn.Parameter(torch.randn(n_pos, d) * 0.02)
        self.key = nn.Linear(d, d)
        # direct recognition energy phi(a_i,h_i)->4
        self.phi = nn.Sequential(nn.Linear(aa_dim + d, d), nn.GELU(), nn.Linear(d, 4))
        # second-shell psi(a_i,a_k,h_i,h_k)->4
        self.psi = nn.Sequential(nn.Linear(2 * aa_dim + 2 * d, d), nn.GELU(), nn.Linear(d, 4))
        self.log_lam_d = nn.Parameter(torch.tensor(0.0))    # exp -> 1.0
        self.log_lam_s = nn.Parameter(torch.tensor(-1.2))   # exp -> ~0.30

    def contact(self, hp, oracle_bias=None):
        """C[j,i] soft attention over residues; oracle_bias (L,) added to logits (soft)."""
        s = self.pos_q @ self.key(hp).T                      # (n_pos, L)
        if oracle_bias is not None:
            s = s + oracle_bias.view(1, -1)
        return F.softmax(s, dim=-1)                          # (n_pos, L)

    def forward(self, h, aa_onehot, fam_id=0, oracle_bias=None,
                direct_on=True, second_shell_on=None, return_parts=False):
        """h:(L,esm_dim) frozen; aa_onehot:(L,aa_dim). Returns z:(n_pos,4) logits."""
        L = h.shape[0]
        hp = self.proj(h)                                    # (L,d)
        C = self.contact(hp, oracle_bias)                    # (n_pos,L)
        z = self.r[fam_id].t().clone()                       # (n_pos,4) from (4,n_pos)
        parts = {}
        if direct_on:
            e_phi = self.phi(torch.cat([aa_onehot, hp], -1)) # (L,4)
            z_direct = C @ e_phi                             # (n_pos,4)
            z = z + self.log_lam_d.exp() * z_direct
            if return_parts:
                parts["direct_per_res"] = e_phi              # (L,4)
                parts["C"] = C
        ss = self.use_second_shell if second_shell_on is None else second_shell_on
        if ss:
            z_ss = torch.zeros(self.n_pos, 4, device=h.device)
            for off in range(1, self.nbhd + 1):
                if off >= L: break
                # ordered pairs (i, k=i+off) and (i, k=i-off), A[i,k]=1/(2*nbhd)
                for ia, ka in ((slice(0, L - off), slice(off, L)),
                               (slice(off, L), slice(0, L - off))):
                    pin = torch.cat([aa_onehot[ia], aa_onehot[ka], hp[ia], hp[ka]], -1)
                    e = self.psi(pin)                        # (L-off,4)
                    z_ss = z_ss + (C[:, ia] @ e)
            z = z + self.log_lam_s.exp() * (z_ss / (2 * self.nbhd))
        if return_parts:
            parts["z"] = z
            return z, parts
        return z                                             # (n_pos,4)
