"""Simulation-based Monte Carlo guidance (g_sim-MC) on SE(3)^N.

Hybrid adaptation of GGPS (arXiv:2502.02150) for SE(3)^N product space:

Original GGPS (single SE(3)):
  - Sample D random priors x_0^(d) ~ p_prior
  - Extrapolate geodesic through x_t to get x_1^(d)
  - Works because 6D space is well-covered by K~8 samples

Problem in SE(3)^N (N residues, 6N dimensions):
  - Random priors give physically meaningless protein candidates
  - K << exp(6N) → importance weights are uniform → no guidance signal

Hybrid solution:
  1. Model predicts rigid_pred (point estimate of x_1)
  2. Backtrack to implied prior: x_0_impl = geodesic_backtrack(x_t, rigid_pred, t)
  3. Perturb x_0_impl with small noise → K candidates x_0^(k)
  4. Extrapolate each x_0^(k) through x_t → x_1^(k)  (original GGPS formula)
  5. Evaluate J, compute importance weights, aggregate velocities

This preserves the original GGPS math (prior → geodesic extrapolation)
while using the model prediction to focus candidates in a relevant region.
The perturbation sigma is on the NOISE side (x_0), not the data side (x_1).
"""

from typing import Callable

import torch

from foldflow.guidance.se3n_utils import (
    geodesic_backtrack,
    geodesic_extrapolate,
    perturb_frames,
    rigids_to_rot_trans,
    velocity_toward,
)


class SimMCGuidance:
    def __init__(
        self,
        energy_fn: Callable,
        K: int = 8,
        sigma_rot: float = 0.3,
        sigma_trans: float = 0.5,
        lambda_: float = 1.0,
        temperature: float = 1.0,
        coordinate_scaling: float = 0.1,
        so3_inference_scaling: float = 10.0,
        eps: float = 1e-3,
        t_extrapolate_eps: float = 0.025,
    ):
        """
        Args:
            energy_fn: callable J(R, x) -> [...] scalar.
            K: number of Monte Carlo candidates per reverse step.
            sigma_rot: perturbation std (rad) on implied x_0 rotations.
            sigma_trans: perturbation std on implied x_0 translations
                (in flow_matcher's internal scale, NOT Angstroms).
            lambda_: guidance strength coefficient.
            temperature: softmax temperature on -J.
            coordinate_scaling / so3_inference_scaling: match flow_matcher units.
            t_extrapolate_eps: clamp for t in geodesic ops.
        """
        self.J = energy_fn
        self.K = K
        self.sigma_rot = sigma_rot
        self.sigma_trans = sigma_trans
        self.lambda_ = lambda_
        self.temperature = temperature
        self.coordinate_scaling = coordinate_scaling
        self.so3_inference_scaling = so3_inference_scaling
        self.eps = eps
        self.t_extrapolate_eps = t_extrapolate_eps

    @torch.no_grad()
    def compute(
        self,
        rigids_t: torch.Tensor,
        rigid_pred: torch.Tensor,
        t: float,
        flow_mask: torch.Tensor,
    ):
        """Compute the SE(3)^N guidance correction.

        Args:
            rigids_t: [B, N, 7] current noisy state.
            rigid_pred: [B, N, 7] model's predicted x_1 (clean data estimate).
            t: scalar continuous time in [0, 1] (FoldFlow++ convention t=1 noise).
            flow_mask: [B, N] 1=scaffold (flowing), 0=motif (fixed).
        Returns:
            g_rot: [B, N, 3, 3] additive correction for rot_vectorfield.
            g_trans: [B, N, 3] additive correction for trans_vectorfield.
            info: dict with diagnostic scalars.
        """
        R_t, x_t = rigids_to_rot_trans(rigids_t)
        R_pred, x_pred = rigids_to_rot_trans(rigid_pred)

        B, N = R_t.shape[0], R_t.shape[1]
        K = self.K
        t_safe = max(float(t), self.eps)

        # Step 1: Backtrack model prediction to implied noise prior
        R_0_impl, x_0_impl = geodesic_backtrack(
            R_t, x_t, R_pred, x_pred, t_safe,
            eps=self.t_extrapolate_eps,
        )
        # R_0_impl: [B, N, 3, 3], x_0_impl: [B, N, 3]

        # Step 2: Perturb implied prior → K candidates x_0^(k)
        # Scale σ by t: extrapolation amplifies perturbations by (1-t)/t,
        # so σ_0 * t gives data-side effect ≈ σ_0 * (1-t), bounded and smooth.
        sigma_rot_t = self.sigma_rot * t_safe
        sigma_trans_t = self.sigma_trans * t_safe
        R_0_k, x_0_k = perturb_frames(
            R_0_impl, x_0_impl, K,
            sigma_rot=sigma_rot_t,
            sigma_trans=sigma_trans_t,
        )
        # R_0_k: [B, K, N, 3, 3], x_0_k: [B, K, N, 3]

        # Step 3: Geodesic extrapolate each x_0^(k) through x_t → x_1^(k)
        R_t_exp = R_t.unsqueeze(1).expand_as(R_0_k)
        x_t_exp = x_t.unsqueeze(1).expand_as(x_0_k)
        R_1_k, x_1_k = geodesic_extrapolate(
            R_0_k, x_0_k, R_t_exp, x_t_exp, t_safe,
            eps=self.t_extrapolate_eps,
        )
        # R_1_k: [B, K, N, 3, 3], x_1_k: [B, K, N, 3]

        # Step 4: Per-candidate velocity from (R_t, x_t) toward (R_1^(k), x_1^(k))
        v_rot_k, v_trans_k = velocity_toward(
            R_t.unsqueeze(1),
            x_t.unsqueeze(1),
            R_1_k,
            x_1_k,
            t_safe,
            coordinate_scaling=self.coordinate_scaling,
            so3_inference_scaling=self.so3_inference_scaling,
        )
        # v_rot_k: [B, K, N, 3, 3]; v_trans_k: [B, K, N, 3]

        # Step 5: Evaluate J on each candidate, importance weights
        J_vals = self.J(R_1_k, x_1_k)  # [B, K]
        if self.temperature != 1.0:
            logits = -J_vals / self.temperature
        else:
            logits = -J_vals
        log_Z = torch.logsumexp(logits, dim=1) - torch.log(
            torch.tensor(float(K), device=J_vals.device, dtype=J_vals.dtype)
        )
        alpha = torch.exp(logits - log_Z.unsqueeze(1)) - 1.0  # [B, K]
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 6: g = mean_k(alpha_k * v_k)
        w = (alpha / K).view(B, K, 1, 1, 1)
        g_rot = (w * v_rot_k).sum(dim=1)  # [B, N, 3, 3]
        w2 = (alpha / K).view(B, K, 1, 1)
        g_trans = (w2 * v_trans_k).sum(dim=1)  # [B, N, 3]

        # Zero guidance on fixed (motif) residues
        fm_rot = flow_mask.unsqueeze(-1).unsqueeze(-1)
        fm_trans = flow_mask.unsqueeze(-1)
        g_rot = g_rot * fm_rot
        g_trans = g_trans * fm_trans

        info = {
            "J_mean": J_vals.mean().item(),
            "J_min": J_vals.min().item(),
            "J_max": J_vals.max().item(),
            "alpha_abs_mean": alpha.abs().mean().item(),
            "g_rot_norm": g_rot.norm().item(),
            "g_trans_norm": g_trans.norm().item(),
        }
        return self.lambda_ * g_rot, self.lambda_ * g_trans, info
