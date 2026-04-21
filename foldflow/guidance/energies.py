"""Test-time energy / cost functions J for guidance."""

from typing import Optional

import torch

from foldflow.guidance.se3n_utils import so3_log_skew


class J_motif:
    """Motif-scaffolding energy on SE(3)^N.

    J(T) = sum_{i in motif} alpha * ||log(R_target_i^T R_i)||_F^2
                         + beta  * ||x_i - x_target_i||^2
           + gamma * clash_term (optional)

    All inputs are torch tensors on the same device.
    """

    def __init__(
        self,
        target_R: torch.Tensor,  # [M, 3, 3]
        target_x: torch.Tensor,  # [M, 3]
        motif_indices: torch.Tensor,  # [M] long tensor (indices in [0,N))
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        clash_radius: float = 3.8,
    ):
        self.target_R = target_R
        self.target_x = target_x
        self.motif_indices = motif_indices
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.clash_radius = clash_radius

    def to(self, device, dtype=None):
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.target_R = self.target_R.to(**kwargs)
        self.target_x = self.target_x.to(**kwargs)
        self.motif_indices = self.motif_indices.to(device=device)
        return self

    def __call__(self, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate J on a batch of SE(3)^N states.

        Args:
            R: [..., N, 3, 3] rotations.
            x: [..., N, 3] translations.
        Returns:
            J: [...] per-sample scalar energy.
        """
        idx = self.motif_indices
        R_m = R.index_select(-3, idx)  # [..., M, 3, 3]
        x_m = x.index_select(-2, idx)  # [..., M, 3]

        tgt_R = self.target_R
        tgt_x = self.target_x
        while tgt_R.dim() < R_m.dim():
            tgt_R = tgt_R.unsqueeze(0)
            tgt_x = tgt_x.unsqueeze(0)

        # SO(3) geodesic^2 via Frobenius norm of log map.
        rel = torch.einsum("...ij,...jk->...ik", tgt_R.transpose(-1, -2), R_m)
        log_rel = so3_log_skew(rel)
        rot_sq = (log_rel ** 2).sum(dim=(-1, -2))  # [..., M]
        trans_sq = ((x_m - tgt_x) ** 2).sum(dim=-1)  # [..., M]

        J = self.alpha * rot_sq.sum(dim=-1) + self.beta * trans_sq.sum(dim=-1)

        if self.gamma > 0.0:
            # Soft clash: penalize non-motif residues that are too close to motif residues.
            # Cheap O(N*M) pairwise; only translations used.
            x_all = x  # [..., N, 3]
            # Build non-motif mask.
            N = x_all.shape[-2]
            mask = torch.ones(N, device=x_all.device, dtype=torch.bool)
            mask[idx] = False  # True = scaffold residue
            x_scaf = x_all[..., mask, :]  # [..., N-M, 3]
            dists = torch.norm(
                x_scaf.unsqueeze(-2) - x_m.unsqueeze(-3), dim=-1
            )  # [..., N-M, M]
            clash = torch.clamp(self.clash_radius - dists, min=0.0) ** 2
            J = J + self.gamma * clash.sum(dim=(-1, -2))

        return J
