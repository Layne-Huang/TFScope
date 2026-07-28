"""Differentiable latent offset/reverse-complement registration losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


RC_ROWS = (3, 2, 1, 0)


def registration_states(max_shift: int) -> tuple[tuple[int, int], ...]:
    """Return (orientation, offset) states; orientation 0=fwd and 1=RC."""
    return tuple(
        (orientation, offset)
        for orientation in (0, 1)
        for offset in range(-max_shift, max_shift + 1)
    )


def transform_target(
    target_pwm: torch.Tensor,
    pwm_mask: torch.Tensor,
    orientation: int,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform one padded target into the model output frame.

    ``offset`` follows the E3 convention: transformed source column ``i`` is
    placed at output column ``i + offset``. Columns outside the output window
    are cropped. The returned coverage is relative to the original valid motif.
    """
    length = target_pwm.shape[-1]
    valid_columns = torch.nonzero(pwm_mask > 0.5, as_tuple=False).flatten()
    motif = target_pwm[:, valid_columns]
    if orientation == 1:
        rows = torch.tensor(RC_ROWS, device=target_pwm.device)
        motif = motif.index_select(0, rows).flip(-1)

    transformed = torch.full_like(target_pwm, 0.25)
    transformed_mask = torch.zeros_like(pwm_mask)
    source_start = max(0, -offset)
    source_end = min(motif.shape[-1], length - offset)
    overlap = max(source_end - source_start, 0)
    if overlap:
        destination_start = source_start + offset
        destination_end = destination_start + overlap
        transformed[:, destination_start:destination_end] = motif[
            :, source_start:source_end
        ]
        transformed_mask[destination_start:destination_end] = 1.0
    coverage = transformed_mask.sum() / max(int(valid_columns.numel()), 1)
    return transformed, transformed_mask, coverage


def enumerate_registered_targets(
    target_pwm: torch.Tensor,
    pwm_mask: torch.Tensor,
    max_shift: int,
    min_overlap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate transformed targets for all offset/RC states.

    Returns targets ``(B,S,4,L)``, masks ``(B,S,L)``, coverage ``(B,S)``, and
    valid-state flags ``(B,S)``.
    """
    states = registration_states(max_shift)
    batch_size, _, length = target_pwm.shape
    targets = target_pwm.new_full((batch_size, len(states), 4, length), 0.25)
    masks = pwm_mask.new_zeros((batch_size, len(states), length))
    coverage = pwm_mask.new_zeros((batch_size, len(states)))
    valid = torch.zeros(
        batch_size, len(states), dtype=torch.bool, device=target_pwm.device
    )
    for batch_index in range(batch_size):
        original_length = int((pwm_mask[batch_index] > 0.5).sum().item())
        required_overlap = min(min_overlap, max(original_length, 1))
        for state_index, (orientation, offset) in enumerate(states):
            transformed, transformed_mask, state_coverage = transform_target(
                target_pwm[batch_index],
                pwm_mask[batch_index],
                orientation,
                offset,
            )
            targets[batch_index, state_index] = transformed
            masks[batch_index, state_index] = transformed_mask
            coverage[batch_index, state_index] = state_coverage
            valid[batch_index, state_index] = (
                transformed_mask.sum() >= required_overlap
            )
    return targets, masks, coverage, valid


def _state_index(
    orientation: torch.Tensor,
    offset: torch.Tensor,
    max_shift: int,
) -> torch.Tensor:
    offsets_per_orientation = 2 * max_shift + 1
    return orientation.long() * offsets_per_orientation + offset.long() + max_shift


def registration_state_index(
    orientation: torch.Tensor,
    offset: torch.Tensor,
    max_shift: int,
) -> torch.Tensor:
    return _state_index(orientation, offset, max_shift)


def allowed_registration_states(
    valid_states: torch.Tensor,
    anchor_mask: torch.Tensor | None,
    anchor_mode: torch.Tensor | None,
    anchor_orientation: torch.Tensor | None,
    anchor_offset: torch.Tensor | None,
    max_shift: int,
) -> torch.Tensor:
    """Restrict anchored samples to an orientation or exact state."""
    allowed = valid_states.clone()
    if anchor_mask is None:
        return allowed
    anchored = anchor_mask > 0.5
    if not anchored.any():
        return allowed
    if anchor_orientation is None or anchor_offset is None:
        raise ValueError("Registration anchors require orientation and offset")
    if anchor_mode is None:
        anchor_mode = torch.full_like(anchor_orientation, 2)
    orientation_only = anchored & (anchor_mode == 1)
    exact_state = anchored & (anchor_mode == 2)
    invalid_mode = anchored & ~((anchor_mode == 1) | (anchor_mode == 2))
    if invalid_mode.any():
        raise ValueError("Registration anchor mode must be 1 or 2")
    if orientation_only.any():
        states_per_orientation = 2 * max_shift + 1
        state_orientations = (
            torch.arange(allowed.shape[1], device=allowed.device)
            // states_per_orientation
        )
        allowed[orientation_only] &= (
            state_orientations.unsqueeze(0)
            == anchor_orientation[orientation_only].unsqueeze(1)
        )
    indices = _state_index(anchor_orientation, anchor_offset, max_shift)
    if ((indices < 0) | (indices >= allowed.shape[1])).any():
        raise ValueError("Registration anchor offset is outside the state range")
    allowed[exact_state] = False
    allowed[exact_state, indices[exact_state]] = True
    if not (allowed[anchored].any(dim=1)).all():
        raise ValueError("Registration anchor has insufficient valid overlap")
    return allowed


def _state_masked_mean(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return (values * masks).sum(dim=-1) / masks.sum(dim=-1).clamp(min=1.0)


def latent_registration_loss(
    pred_gate: torch.Tensor,
    pred_pwm: torch.Tensor,
    target_pwm: torch.Tensor,
    pwm_mask: torch.Tensor,
    *,
    max_shift: int,
    min_overlap: int,
    temperature: float,
    coverage_penalty: float,
    gate_weight: float,
    l1_weight: float,
    ic_weight: float,
    ic_pcc_weight: float,
    topbase_weight: float,
    topbase_margin: float,
    topbase_ic_thresh_bits: float,
    anchor_mask: torch.Tensor | None = None,
    anchor_mode: torch.Tensor | None = None,
    anchor_orientation: torch.Tensor | None = None,
    anchor_offset: torch.Tensor | None = None,
    register_logits: torch.Tensor | None = None,
    register_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    """Marginalize the supervised loss over valid registration states."""
    if temperature <= 0:
        raise ValueError("registration temperature must be positive")
    targets, masks, coverage, valid = enumerate_registered_targets(
        target_pwm, pwm_mask, max_shift, min_overlap
    )
    allowed = allowed_registration_states(
        valid,
        anchor_mask,
        anchor_mode,
        anchor_orientation,
        anchor_offset,
        max_shift,
    )
    if not allowed.any(dim=1).all():
        raise ValueError("At least one sample has no valid registration state")

    pred_log_probs = F.log_softmax(pred_pwm, dim=1)
    pred_probs = pred_log_probs.exp()
    expanded_probs = pred_probs.unsqueeze(1)
    expanded_log_probs = pred_log_probs.unsqueeze(1)

    l1_per_position = (expanded_probs - targets).abs().sum(dim=2)
    state_l1 = _state_masked_mean(l1_per_position, masks)

    log4 = math.log(4.0)
    clipped_target = targets.clamp(1e-8, 1.0)
    target_ic = (clipped_target * clipped_target.log()).sum(dim=2) + log4
    pred_ic = (
        expanded_probs * expanded_log_probs
    ).sum(dim=2) + log4
    state_ic = _state_masked_mean((target_ic - pred_ic).abs(), masks)

    pred_centered = expanded_probs - expanded_probs.mean(dim=2, keepdim=True)
    target_centered = targets - targets.mean(dim=2, keepdim=True)
    correlation = (pred_centered * target_centered).sum(dim=2) / (
        pred_centered.norm(dim=2) * target_centered.norm(dim=2) + 1e-8
    )
    ic_weights = target_ic.clamp(min=0.0) * masks
    ic_weights = ic_weights / ic_weights.sum(dim=-1, keepdim=True).clamp(
        min=1e-8
    )
    state_ic_pcc = ((1.0 - correlation) * ic_weights).sum(dim=-1)

    true_top = targets.argmax(dim=2)
    expanded_logits = pred_pwm.unsqueeze(1).expand(-1, targets.shape[1], -1, -1)
    true_logits = expanded_logits.gather(
        2, true_top.unsqueeze(2)
    ).squeeze(2)
    negative_logits = expanded_logits.clone()
    negative_logits.scatter_(2, true_top.unsqueeze(2), float("-inf"))
    runner_up = negative_logits.max(dim=2).values
    high_ic = (
        target_ic > topbase_ic_thresh_bits * math.log(2.0)
    ).to(masks.dtype) * masks
    state_topbase = (
        F.relu(topbase_margin - (true_logits - runner_up)) * high_ic
    ).sum(dim=-1) / high_ic.sum(dim=-1).clamp(min=1.0)

    gate_targets = masks
    gate_logits = pred_gate.unsqueeze(1).expand_as(gate_targets)
    state_gate = F.binary_cross_entropy_with_logits(
        gate_logits, gate_targets, reduction="none"
    ).mean(dim=-1)

    state_pwm = (
        l1_weight * state_l1
        + ic_weight * state_ic
        + ic_pcc_weight * state_ic_pcc
        + topbase_weight * state_topbase
    )
    state_total = (
        state_pwm
        + gate_weight * state_gate
        + coverage_penalty * (1.0 - coverage)
    )
    state_total = state_total.masked_fill(~allowed, float("inf"))

    if register_logits is not None:
        if register_logits.shape != state_total.shape:
            raise ValueError(
                "register_logits must match the enumerated registration states"
            )
        log_prior = F.log_softmax(register_logits, dim=1)
        log_prior = log_prior.masked_fill(~allowed, float("-inf"))
        log_prior = log_prior - torch.logsumexp(log_prior, dim=1, keepdim=True)
    else:
        log_prior = -allowed.sum(dim=1, keepdim=True).to(state_total.dtype).log()
        log_prior = log_prior.masked_fill(~allowed, float("-inf"))
    log_weights = log_prior - state_total / temperature
    marginal = -temperature * torch.logsumexp(log_weights, dim=1)
    posterior = torch.softmax(log_weights, dim=1)
    posterior_detached = posterior.detach()
    expected_mask = (posterior_detached.unsqueeze(-1) * masks).sum(dim=1)
    expected_target = (
        posterior_detached.unsqueeze(-1).unsqueeze(-1) * targets
    ).sum(dim=1)

    def expected(values):
        return (posterior_detached * values).sum(dim=1).mean()

    register_supervision = pred_pwm.new_zeros(())
    register_accuracy = pred_pwm.new_zeros(())
    if (
        register_logits is not None
        and anchor_mask is not None
        and (anchor_mask > 0.5).any()
    ):
        anchored = anchor_mask > 0.5
        if anchor_mode is None:
            anchor_mode = torch.full_like(anchor_orientation, 2)
        anchor_indices = _state_index(
            anchor_orientation, anchor_offset, max_shift
        )
        supervision_terms = []
        accuracy_terms = []
        exact_state = anchored & (anchor_mode == 2)
        if exact_state.any():
            supervision_terms.append(
                F.cross_entropy(
                    register_logits[exact_state],
                    anchor_indices[exact_state],
                )
            )
            accuracy_terms.append(
                (
                    register_logits[exact_state].argmax(dim=1)
                    == anchor_indices[exact_state]
                ).float()
            )
        orientation_only = anchored & (anchor_mode == 1)
        if orientation_only.any():
            states_per_orientation = 2 * max_shift + 1
            orientation_logits = torch.stack(
                [
                    torch.logsumexp(
                        register_logits[
                            orientation_only,
                            orientation * states_per_orientation
                            : (orientation + 1) * states_per_orientation,
                        ],
                        dim=1,
                    )
                    for orientation in (0, 1)
                ],
                dim=1,
            )
            supervision_terms.append(
                F.cross_entropy(
                    orientation_logits,
                    anchor_orientation[orientation_only],
                )
            )
            accuracy_terms.append(
                (
                    orientation_logits.argmax(dim=1)
                    == anchor_orientation[orientation_only]
                ).float()
            )
        register_supervision = torch.stack(supervision_terms).mean()
        register_accuracy = torch.cat(accuracy_terms).mean()
    total_loss = marginal.mean() + register_loss_weight * register_supervision
    metrics = {
        "registration_loss": marginal.mean(),
        "registration_gate": expected(state_gate),
        "registration_pwm": expected(state_pwm),
        "registration_l1": expected(state_l1),
        "registration_ic": expected(state_ic),
        "registration_ic_pcc": expected(state_ic_pcc),
        "registration_topbase": expected(state_topbase),
        "registration_coverage": expected(coverage),
        "registration_entropy": (
            -(posterior * posterior.clamp_min(1e-8).log()).sum(dim=1).mean()
        ),
        "registration_anchor_fraction": (
            (anchor_mask > 0.5).float().mean()
            if anchor_mask is not None
            else pred_pwm.new_zeros(())
        ),
        "register_supervision": register_supervision,
        "register_accuracy": register_accuracy,
    }
    return total_loss, metrics, expected_target, expected_mask


def export_registered_predictions(
    gate_logits: torch.Tensor,
    pwm_logits: torch.Tensor,
    register_logits: torch.Tensor,
    max_shift: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Export internal-frame predictions to left-anchored canonical coordinates."""
    probabilities = torch.softmax(pwm_logits, dim=1)
    batch_size, _, length = probabilities.shape
    output_probabilities = torch.full_like(probabilities, 0.25)
    output_gates = torch.full_like(gate_logits, -8.0)
    state_indices = register_logits.argmax(dim=1)
    states_per_orientation = 2 * max_shift + 1
    orientations = state_indices // states_per_orientation

    for batch_index in range(batch_size):
        active = gate_logits[batch_index].sigmoid() > 0.5
        if not active.any():
            estimated_length = int(
                gate_logits[batch_index].sigmoid().sum().round().clamp(1, length)
            )
            active_indices = torch.topk(
                gate_logits[batch_index], estimated_length
            ).indices.sort().values
        else:
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
        start = int(active_indices.min())
        end = int(active_indices.max()) + 1
        motif = probabilities[batch_index, :, start:end]
        motif_gate = gate_logits[batch_index, start:end]
        if int(orientations[batch_index]) == 1:
            rows = torch.tensor(RC_ROWS, device=pwm_logits.device)
            motif = motif.index_select(0, rows).flip(-1)
            motif_gate = motif_gate.flip(-1)
        motif_length = min(motif.shape[-1], length)
        output_probabilities[batch_index, :, :motif_length] = motif[
            :, :motif_length
        ]
        output_gates[batch_index, :motif_length] = motif_gate[:motif_length]
    return output_gates, output_probabilities.clamp_min(1e-8).log()
