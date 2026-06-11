import torch
import torch.nn.functional as F


def load_balance_loss(gate_logits: torch.Tensor, top_indices: torch.Tensor,
                      num_experts: int = 12, alpha: float = 0.01) -> torch.Tensor:
    """Switch Transformer style load balance loss."""
    B = gate_logits.shape[0]
    expert_mask = torch.zeros(B, num_experts, device=gate_logits.device)
    for k in range(top_indices.shape[1]):
        expert_mask.scatter_(1, top_indices[:, k:k+1], 1.0)
    f = expert_mask.mean(dim=0)
    P = F.softmax(gate_logits, dim=-1).mean(dim=0)
    return alpha * num_experts * (f * P).sum()


def family_diversity_loss(gate_logits: torch.Tensor, family_id: torch.Tensor,
                          num_families: int = 10, num_experts: int = 12,
                          alpha: float = 0.005) -> torch.Tensor:
    """Encourage routing diversity within each family."""
    probs = F.softmax(gate_logits, dim=-1)
    loss = torch.tensor(0.0, device=gate_logits.device)
    count = 0
    for f in range(num_families):
        mask = (family_id == f)
        if mask.sum() > 1:
            mean_probs = probs[mask].mean(dim=0)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
            loss = loss - entropy
            count += 1
    if count > 0:
        loss = loss / count
    return alpha * loss
