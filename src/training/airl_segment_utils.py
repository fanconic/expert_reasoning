"""Utilities for segment-level AIRL reward and advantage computation.

The helpers here are intentionally tensor-oriented and independent from the
trainer so segment boundary logic can be tested without loading models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SegmentLayout:
    """Padded segment index tensors for a batch.

    All indices refer to positions in the tokenized full prompt+completion
    sequence. Invalid padded segment slots have `valid_mask == False`.
    """

    starts: torch.Tensor
    ends: torch.Tensor
    prev_indices: torch.Tensor
    next_indices: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def max_segments(self) -> int:
        return int(self.valid_mask.shape[1])


def fixed_interval_boundary_mask(completion_mask: torch.Tensor, segment_tokens: int) -> torch.Tensor:
    """Mark every `segment_tokens` completion tokens and always mark the last token."""
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive.")

    completion_mask = completion_mask.bool()
    token_positions = completion_mask.long().cumsum(dim=1)
    boundary_mask = (token_positions % int(segment_tokens) == 0) & completion_mask

    lengths = completion_mask.long().sum(dim=1)
    if lengths.any():
        positions = torch.arange(completion_mask.size(1), device=completion_mask.device)
        last_indices = (completion_mask.long() * positions.unsqueeze(0)).max(dim=1).values
        rows = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        boundary_mask[rows, last_indices[rows]] = True

    return boundary_mask


def segment_layout_from_boundaries(
    boundary_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> SegmentLayout:
    """Build padded segment start/end/prefix indices from completion boundaries."""
    boundary_mask = boundary_mask.bool() & completion_mask.bool()
    completion_mask = completion_mask.bool()
    device = completion_mask.device

    per_row = []
    max_segments = 0
    for b in range(completion_mask.size(0)):
        completion_positions = torch.nonzero(completion_mask[b], as_tuple=False).flatten()
        if completion_positions.numel() == 0:
            row_segments = []
        else:
            boundaries = torch.nonzero(boundary_mask[b], as_tuple=False).flatten()
            if boundaries.numel() == 0 or boundaries[-1] != completion_positions[-1]:
                boundaries = torch.cat([boundaries, completion_positions[-1:]])

            row_segments = []
            start = int(completion_positions[0].item())
            for end_tensor in boundaries:
                end = int(end_tensor.item())
                if end < start:
                    continue
                row_segments.append((start, end))
                start = end + 1
                if start > int(completion_positions[-1].item()):
                    break

        per_row.append(row_segments)
        max_segments = max(max_segments, len(row_segments))

    max_segments = max(max_segments, 1)
    shape = (completion_mask.size(0), max_segments)
    starts = torch.zeros(shape, dtype=torch.long, device=device)
    ends = torch.zeros(shape, dtype=torch.long, device=device)
    prev_indices = torch.zeros(shape, dtype=torch.long, device=device)
    next_indices = torch.zeros(shape, dtype=torch.long, device=device)
    valid_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    for b, row_segments in enumerate(per_row):
        for s, (start, end) in enumerate(row_segments):
            starts[b, s] = start
            ends[b, s] = end
            prev_indices[b, s] = max(start - 1, 0)
            next_indices[b, s] = end
            valid_mask[b, s] = True

    return SegmentLayout(
        starts=starts,
        ends=ends,
        prev_indices=prev_indices,
        next_indices=next_indices,
        valid_mask=valid_mask,
    )


def gather_token_positions(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather token positions from `[B, L]` or `[B, L, D]` tensors with `[B, S]` indices."""
    if sequence.ndim == 2:
        return sequence.gather(1, indices)
    if sequence.ndim == 3:
        expanded = indices.unsqueeze(-1).expand(-1, -1, sequence.size(-1))
        return sequence.gather(1, expanded)
    raise ValueError(f"Expected sequence rank 2 or 3, got shape {tuple(sequence.shape)}.")


def sum_tokens_by_segment(
    token_values: torch.Tensor,
    layout: SegmentLayout,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Sum `[B, L]` token values inside each segment described by `layout`."""
    if token_values.ndim != 2:
        raise ValueError(f"Expected token_values rank 2, got shape {tuple(token_values.shape)}.")

    out = torch.full(
        layout.valid_mask.shape,
        fill_value=float(fill_value),
        dtype=token_values.dtype,
        device=token_values.device,
    )
    for b in range(token_values.size(0)):
        for s in torch.nonzero(layout.valid_mask[b], as_tuple=False).flatten().tolist():
            out[b, s] = token_values[b, layout.starts[b, s] : layout.ends[b, s] + 1].sum()
    return out


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Mean over valid entries, returning zero where no entries are valid."""
    mask = mask.to(dtype=values.dtype, device=values.device)
    summed = (values * mask).sum(dim=dim)
    count = mask.sum(dim=dim).clamp_min(1.0)
    return summed / count


def masked_std(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Population std over valid entries, returning zero where no entries are valid."""
    mean = masked_mean(values, mask, dim=dim).unsqueeze(dim)
    mask_f = mask.to(dtype=values.dtype, device=values.device)
    count = mask_f.sum(dim=dim).clamp_min(1.0)
    var = (((values - mean) ** 2) * mask_f).sum(dim=dim) / count
    return torch.sqrt(var.clamp_min(0.0))


def normalize_segments_by_group(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    group_size: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Group-normalize valid segment values across rollout groups.

    Normalization is over all valid intervals in a GRPO group. This is useful
    when policy advantages are interval-dense rather than sequence-level.
    """
    if values.shape != valid_mask.shape:
        raise ValueError(
            f"values and valid_mask must have the same shape: {tuple(values.shape)} != {tuple(valid_mask.shape)}"
        )
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    out = torch.zeros_like(values)
    for start in range(0, values.size(0), int(group_size)):
        end = min(start + int(group_size), values.size(0))
        mask = valid_mask[start:end]
        if not mask.any():
            continue
        block_values = values[start:end]
        valid_values = block_values[mask]
        std = valid_values.std(unbiased=False).clamp_min(float(eps))
        block_out = out[start:end]
        block_out[mask] = (valid_values - valid_values.mean()) / std
        out[start:end] = block_out
    return out


def broadcast_segment_values_to_tokens(
    segment_values: torch.Tensor,
    layout: SegmentLayout,
    seq_len: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Assign each segment value to the completion tokens inside that segment."""
    if segment_values.shape != layout.valid_mask.shape:
        raise ValueError(
            "segment_values shape must match layout.valid_mask shape: "
            f"{tuple(segment_values.shape)} != {tuple(layout.valid_mask.shape)}"
        )

    out = torch.full(
        (segment_values.size(0), int(seq_len)),
        fill_value=float(fill_value),
        dtype=segment_values.dtype,
        device=segment_values.device,
    )
    for b in range(segment_values.size(0)):
        for s in torch.nonzero(layout.valid_mask[b], as_tuple=False).flatten().tolist():
            out[b, layout.starts[b, s] : layout.ends[b, s] + 1] = segment_values[b, s]
    return out


def assert_all_finite(name: str, value: torch.Tensor) -> None:
    """Raise a compact error if a diagnostic tensor contains NaNs or infinities."""
    if not torch.isfinite(value).all():
        bad = (~torch.isfinite(value)).sum().item()
        raise FloatingPointError(f"{name} contains {bad} non-finite values.")
