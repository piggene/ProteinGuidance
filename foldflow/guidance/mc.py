"""Reference-trajectory Monte Carlo guidance (g_MC) on SE(3)^N.

Adapted from GGPS guidance_vector_MC. Instead of a ground-truth grasp pose bank,
we use a bank of natural protein backbones of length N as reference candidates
for x_1. Each residue is treated independently (the product-space assumption is
already implicit in the per-residue velocity formulation).

At reverse step t we:
  1. Load a (pre-cached) bank of D reference SE(3)^N backbones of length N.
  2. Compute per-residue "likelihood" weights w_d that x_t is consistent with
     reference d: using the same SE(3) tangent-residual formulation as GGPS but
     summed over the N residues (log-likelihood is additive).
  3. For each reference d, compute the forward velocity from x_t to x_1^(d) that
     would complete the path.
  4. Compute importance-weighted guidance:
     g_t = sum_d (alpha_d * w_d) * v_t^(d)
     where alpha_d = K*p_J(d)/p_uniform(d) - 1 using J(x_1^(d)) on the full
     backbone.
"""

from typing import Callable, Optional

import torch

from foldflow.guidance.se3n_utils import (
    so3_log_skew,
    rigids_to_rot_trans,
    velocity_toward,
)


class MCGuidance:
    def __init__(
        self,
        reference_bank: dict,
        energy_fn: Callable,
        lambda_: float = 1.0,
        sigma_w: float = 0.3,
        sigma_v: float = 1.0,
        temperature: float = 1.0,
        max_refs_per_call: int = 256,
        coordinate_scaling: float = 0.1,
        so3_inference_scaling: float = 10.0,
        eps: float = 1e-3,
    ):
        """
        Args:
            reference_bank: dict with keys:
              "R_1": [D, N, 3, 3] SO(3) frames of natural backbones of length N.
              "x_1": [D, N, 3] translations (Cα positions, centered).
            energy_fn: J(R, x) -> [...] scalar.
            sigma_w, sigma_v: likelihood stds for (angular, linear) residuals
              in the tangent-space match used in GGPS MC.
            max_refs_per_call: subsample at most this many references per step
              for memory.
        """
        self.ref = reference_bank
        self.J = energy_fn
        self.lambda_ = lambda_
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.temperature = temperature
        self.max_refs_per_call = max_refs_per_call
        self.coordinate_scaling = coordinate_scaling
        self.so3_inference_scaling = so3_inference_scaling
        self.eps = eps

    @torch.no_grad()
    def compute(
        self,
        rigids_t: torch.Tensor,
        rigid_pred: torch.Tensor,
        t: float,
        flow_mask: torch.Tensor,
    ):
        R_t, x_t = rigids_to_rot_trans(rigids_t)  # [B, N, 3, 3], [B, N, 3]
        B, N = R_t.shape[0], R_t.shape[1]
        t_safe = max(float(t), self.eps)

        R_bank = self.ref["R_1"].to(R_t.device, R_t.dtype)  # [D, N, 3, 3]
        x_bank = self.ref["x_1"].to(x_t.device, x_t.dtype)  # [D, N, 3]
        D = R_bank.shape[0]
        if D > self.max_refs_per_call:
            idx = torch.randperm(D, device=R_bank.device)[: self.max_refs_per_call]
            R_bank = R_bank[idx]
            x_bank = x_bank[idx]
            D = self.max_refs_per_call

        # Build reference midpoint (R_t_mean, x_t_mean) = geodesic(x_0_d, x_1_d, t).
        # Without an explicit x_0 we approximate x_t_mean on the geodesic from
        # identity / COM-zero prior to x_1_d at time t: on SE(3) this reduces to
        # R_t_mean = exp(t * log(x_1_d.R)) on the SO(3) factor, and
        # x_t_mean = t * x_1_d.x on the R^3 factor (prior mean 0).
        log_R1 = so3_log_skew(R_bank)  # [D, N, 3, 3]
        R_t_mean = torch.linalg.matrix_exp(t_safe * log_R1)  # [D, N, 3, 3]
        x_t_mean = t_safe * x_bank  # [D, N, 3]

        # Residuals: delta_R = R_t_mean^T R_t  => skew, delta_x = x_t - x_t_mean.
        # Broadcast: R_t [B,N,3,3] vs R_t_mean [D,N,3,3] -> [B,D,N,3,3]
        R_t_exp = R_t.unsqueeze(1)  # [B, 1, N, 3, 3]
        R_mean_exp = R_t_mean.unsqueeze(0)  # [1, D, N, 3, 3]
        rel = torch.einsum(
            "...ij,...jk->...ik", R_mean_exp.transpose(-1, -2), R_t_exp
        )
        log_rel = so3_log_skew(rel)  # [B, D, N, 3, 3] skew
        w_b = torch.stack(
            [log_rel[..., 2, 1], log_rel[..., 0, 2], log_rel[..., 1, 0]], dim=-1
        )  # [B, D, N, 3] rotvec
        v_b = x_t_exp_res = x_t.unsqueeze(1) - x_t_mean.unsqueeze(0)  # [B, D, N, 3]

        # Log-likelihood summed over residues (masked by flow_mask so only
        # scaffold residues contribute to the match; motif residues are equal by
        # construction when we plug in x_target).
        fm = flow_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        ll = -0.5 * (
            ((w_b ** 2) / (self.sigma_w ** 2)).sum(dim=-1)
            + ((v_b ** 2) / (self.sigma_v ** 2)).sum(dim=-1)
        )  # [B, D, N]
        ll = (ll * flow_mask.unsqueeze(1)).sum(dim=-1)  # [B, D]
        w_match = torch.softmax(ll, dim=-1)  # [B, D]

        # Energy weights on full backbone candidates.
        # J expects [..., N, ...] per-sample; broadcast bank against batch.
        R_bank_b = R_bank.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B,D,N,3,3]
        x_bank_b = x_bank.unsqueeze(0).expand(B, -1, -1, -1)  # [B,D,N,3]
        J_vals = self.J(R_bank_b, x_bank_b)  # [B, D]
        if self.temperature != 1.0:
            logits = -J_vals / self.temperature
        else:
            logits = -J_vals
        log_Z = torch.logsumexp(logits, dim=1) - torch.log(
            torch.tensor(float(D), device=J_vals.device, dtype=J_vals.dtype)
        )
        alpha = torch.exp(logits - log_Z.unsqueeze(1)) - 1.0  # [B, D]
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-candidate forward velocity from (R_t, x_t) toward (R_1^(d), x_1^(d)).
        x_t_exp_b = x_t.unsqueeze(1)  # [B, 1, N, 3]
        v_rot_d, v_trans_d = velocity_toward(
            R_t_exp.expand(-1, D, -1, -1, -1),
            x_t_exp_b.expand(-1, D, -1, -1),
            R_bank_b,
            x_bank_b,
            t_safe,
            coordinate_scaling=self.coordinate_scaling,
            so3_inference_scaling=self.so3_inference_scaling,
        )  # [B, D, N, 3, 3], [B, D, N, 3]

        # Guidance: sum_d (alpha_d * w_match_d) * v_t^(d).
        coeff = (alpha * w_match).view(B, D, 1, 1, 1)
        g_rot = (coeff * v_rot_d).sum(dim=1)
        coeff2 = (alpha * w_match).view(B, D, 1, 1)
        g_trans = (coeff2 * v_trans_d).sum(dim=1)

        fm_rot = flow_mask.unsqueeze(-1).unsqueeze(-1)
        fm_trans = flow_mask.unsqueeze(-1)
        g_rot = g_rot * fm_rot
        g_trans = g_trans * fm_trans

        info = {
            "J_mean": J_vals.mean().item(),
            "J_min": J_vals.min().item(),
            "D": int(D),
            "alpha_abs_mean": alpha.abs().mean().item(),
            "w_match_max": w_match.max().item(),
            "g_rot_norm": g_rot.norm().item(),
            "g_trans_norm": g_trans.norm().item(),
        }
        return self.lambda_ * g_rot, self.lambda_ * g_trans, info
