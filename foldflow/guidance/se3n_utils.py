"""SE(3)^N geometry helpers for test-time guidance."""

import torch

from foldflow.utils.so3_helpers import log as _so3_log_to_skew_flat


def so3_log_skew(R: torch.Tensor) -> torch.Tensor:
    """Map SO(3) -> so(3) skew-symmetric matrices with arbitrary leading dims.

    Args:
        R: [..., 3, 3] rotation matrices.
    Returns:
        [..., 3, 3] skew-symmetric log.
    """
    orig_shape = R.shape
    R_flat = R.reshape(-1, 3, 3)
    skew_flat = _so3_log_to_skew_flat(R_flat.double()).to(R.dtype)
    return skew_flat.reshape(orig_shape)


def skew_from_vec(v: torch.Tensor) -> torch.Tensor:
    """Hat operator on arbitrary leading dims.

    Args:
        v: [..., 3]
    Returns:
        [..., 3, 3] skew-symmetric.
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    row0 = torch.stack([O, -z, y], dim=-1)
    row1 = torch.stack([z, O, -x], dim=-1)
    row2 = torch.stack([-y, x, O], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def so3_exp_skew(skew: torch.Tensor) -> torch.Tensor:
    """Exponential map so(3) -> SO(3) via matrix_exp.

    Args:
        skew: [..., 3, 3]
    Returns:
        [..., 3, 3] rotation matrix.
    """
    return torch.linalg.matrix_exp(skew)


def velocity_toward(
    R_t: torch.Tensor,
    x_t: torch.Tensor,
    R_data: torch.Tensor,
    x_data: torch.Tensor,
    t: float,
    coordinate_scaling: float = 0.1,
    so3_inference_scaling: float = 10.0,
    eps: float = 1e-3,
):
    """Forward-direction velocity at (R_t, x_t) on a geodesic originating at
    (R_data, x_data) (t=0 endpoint in FoldFlow++ convention, i.e. clean data).

    Matches the units of the model's `rot_vectorfield` / `trans_vectorfield`:
      SO3FM.vectorfield: u_rot = R_t @ log(R_data^T R_t) * so3_inference_scaling
        (inference_scaling>=0 branch).
      R3FM.vectorfield with scale=True:
        u_trans = (scale(x_t) - scale(x_data)) / (t + eps)
                = coordinate_scaling * (x_t - x_data) / (t + eps)

    Args:
        R_t: [..., N, 3, 3] current rotations.
        x_t: [..., N, 3] current translations (Angstrom).
        R_data: [..., N, 3, 3] candidate clean-data rotations.
        x_data: [..., N, 3] candidate clean-data translations (Angstrom).
        t: FoldFlow++ continuous time in [0,1] (1=noise, 0=data).
        coordinate_scaling: matches r3_fm coordinate_scaling (default 0.1).
        so3_inference_scaling: matches so3_fm inference_scaling (default 10.0).
    Returns:
        v_rot: [..., N, 3, 3]
        v_trans: [..., N, 3]
    """
    ts = max(float(t), eps)
    rel = torch.einsum("...ij,...jk->...ik", R_data.transpose(-1, -2), R_t)
    skew = so3_log_skew(rel)
    v_rot = torch.einsum("...ij,...jk->...ik", R_t, skew) * so3_inference_scaling
    v_trans = coordinate_scaling * (x_t - x_data) / ts
    return v_rot, v_trans


def perturb_frames(
    R: torch.Tensor,
    x: torch.Tensor,
    K: int,
    sigma_rot: float,
    sigma_trans: float,
    generator: torch.Generator = None,
):
    """Create K stochastic candidates by perturbing a batch of frames.

    Args:
        R: [B, N, 3, 3] mean rotation (e.g. model's x_1 prediction).
        x: [B, N, 3] mean translation.
        K: number of candidates.
        sigma_rot: rotvec std (rad) on SO(3).
        sigma_trans: translation std (Angstrom, on the model's internal scale).
    Returns:
        R_k: [B, K, N, 3, 3]
        x_k: [B, K, N, 3]
    """
    B, N = R.shape[0], R.shape[1]
    dev, dtype = R.device, R.dtype
    if generator is None:
        eps_rot = torch.randn(B, K, N, 3, device=dev, dtype=dtype)
        eps_trans = torch.randn(B, K, N, 3, device=dev, dtype=dtype)
    else:
        eps_rot = torch.randn(B, K, N, 3, device=dev, dtype=dtype, generator=generator)
        eps_trans = torch.randn(
            B, K, N, 3, device=dev, dtype=dtype, generator=generator
        )
    rotvec = sigma_rot * eps_rot
    dR = so3_exp_skew(skew_from_vec(rotvec))  # [B, K, N, 3, 3]
    R_exp = R.unsqueeze(1).expand(B, K, N, 3, 3)
    R_k = torch.einsum("bknij,bknjl->bknil", R_exp, dR)
    x_k = x.unsqueeze(1) + sigma_trans * eps_trans
    return R_k, x_k


def geodesic_backtrack(
    R_t: torch.Tensor,
    x_t: torch.Tensor,
    R_1: torch.Tensor,
    x_1: torch.Tensor,
    t: float,
    eps: float = 0.025,
):
    """Given (R_t, x_t) and endpoint (R_1, x_1), recover the noise origin x_0.

    Geodesic: x_t = x_0 · exp(t · log(x_0^{-1} x_1))
    Equivalently: R_t^T R_1 = exp((1-t) · Ω)  where Ω = log(R_0^T R_1)
    So: Ω = log(R_t^T R_1) / (1-t)
    And: R_0 = R_t · exp(-t · Ω) = R_t · exp(-t/(1-t) · log(R_t^T R_1))
    Translation: x_0 = (x_t - t · x_1) / (1 - t)

    Args:
        R_t: [..., N, 3, 3]
        x_t: [..., N, 3]
        R_1: [..., N, 3, 3] model-predicted clean rotations.
        x_1: [..., N, 3] model-predicted clean translations.
        t: scalar time in (0, 1).
    Returns:
        R_0: [..., N, 3, 3] implied noise-prior rotations.
        x_0: [..., N, 3] implied noise-prior translations.
    """
    t_safe = max(float(t), eps)
    one_minus_t = max(1.0 - t_safe, eps)

    # SO(3): R_0 = R_t · exp( -t/(1-t) · log(R_t^T R_1) )
    rel = torch.einsum("...ij,...jk->...ik", R_t.transpose(-1, -2), R_1)
    log_rel = so3_log_skew(rel)  # [..., N, 3, 3]
    R_0 = torch.einsum(
        "...ij,...jk->...ik",
        R_t,
        torch.linalg.matrix_exp(-t_safe / one_minus_t * log_rel),
    )

    # R^3: x_0 = (x_t - t * x_1) / (1 - t)
    x_0 = (x_t - t_safe * x_1) / one_minus_t

    return R_0, x_0


def sample_prior(B: int, K: int, N: int, device, dtype=torch.float32):
    """Sample K independent noise priors per batch element.

    SO(3): Uniform (Haar measure) via scipy.
    R^3: N(0, I).

    Returns:
        R_0: [B, K, N, 3, 3]
        x_0: [B, K, N, 3]
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    total = B * K * N
    R_np = ScipyRot.random(total).as_matrix()  # [total, 3, 3]
    R_0 = torch.tensor(R_np, device=device, dtype=dtype).reshape(B, K, N, 3, 3)
    x_0 = torch.randn(B, K, N, 3, device=device, dtype=dtype)
    return R_0, x_0


def geodesic_extrapolate(
    R_0: torch.Tensor,
    x_0: torch.Tensor,
    R_t: torch.Tensor,
    x_t: torch.Tensor,
    t: float,
    eps: float = 0.025,
):
    """Extrapolate the geodesic x_0 -> x_t to t=1 (clean data endpoint).

    On the geodesic:  x_t = x_0 · exp(t · log(x_0^{-1} x_t))
    Therefore:         x_1 = x_0 · exp(log(x_0^{-1} x_t) / t)

    Args:
        R_0: [..., N, 3, 3] noise-prior rotations.
        x_0: [..., N, 3] noise-prior translations.
        R_t: [..., N, 3, 3] current state rotations (will be broadcast).
        x_t: [..., N, 3] current state translations (will be broadcast).
        t: scalar time in (0, 1].
        eps: clamp for t to avoid division by zero.
    Returns:
        R_1: [..., N, 3, 3] extrapolated clean rotations.
        x_1: [..., N, 3] extrapolated clean translations.
    """
    t_safe = max(float(t), eps)

    # SO(3): R_1 = R_0 @ exp( log(R_0^T R_t) / t )
    rel = torch.einsum("...ij,...jk->...ik", R_0.transpose(-1, -2), R_t)
    log_rel = so3_log_skew(rel)  # [..., N, 3, 3] skew
    R_1 = torch.einsum(
        "...ij,...jk->...ik",
        R_0,
        torch.linalg.matrix_exp(log_rel / t_safe),
    )

    # R^3: x_1 = x_0 + (x_t - x_0) / t
    x_1 = x_0 + (x_t - x_0) / t_safe

    return R_1, x_1


def renoise_frames(
    rigids_clean: torch.Tensor,
    t_target: float,
    flow_mask: torch.Tensor = None,
    motif_rigids: torch.Tensor = None,
) -> torch.Tensor:
    """Re-noise clean rigids to time t_target for iterative refinement.

    Implements SDEdit-style re-noising on SE(3)^N:
      SO(3): R_t = R_clean @ exp(t * log(R_clean^T @ R_noise))
      R^3:   x_t = (1-t)*x_clean + t*x_noise

    Args:
        rigids_clean: [N, 7] or [B, N, 7] clean rigids from previous round.
        t_target: time to re-noise to (0=clean, 1=full noise).
        flow_mask: [N] or [B, N], 1=scaffold (re-noise), 0=motif (keep).
            If None, all residues are re-noised.
        motif_rigids: [N, 7] or [B, N, 7] motif target rigids to plant at
            motif positions. If None, motif positions keep their current values.

    Returns:
        rigids_t: same shape as rigids_clean, re-noised to t_target.
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    squeeze = rigids_clean.ndim == 2
    if squeeze:
        rigids_clean = rigids_clean.unsqueeze(0)

    B, N, _ = rigids_clean.shape
    device = rigids_clean.device
    dtype = rigids_clean.dtype

    R_clean, x_clean = rigids_to_rot_trans(rigids_clean)  # [B, N, 3, 3], [B, N, 3]

    # Sample noise
    R_noise_np = ScipyRot.random(B * N).as_matrix().reshape(B, N, 3, 3)
    R_noise = torch.tensor(R_noise_np, device=device, dtype=torch.float64)
    x_noise = torch.randn(B, N, 3, device=device, dtype=dtype)

    # SO(3) geodesic interpolation: R_t = R_clean @ exp(t * log(R_clean^T @ R_noise))
    R_clean_d = R_clean.double()
    rel = torch.einsum("...ij,...jk->...ik", R_clean_d.transpose(-1, -2), R_noise)
    log_rel = so3_log_skew(rel)
    R_t = torch.einsum(
        "...ij,...jk->...ik",
        R_clean_d,
        torch.linalg.matrix_exp(t_target * log_rel),
    ).to(dtype)

    # R^3 linear interpolation: x_t = (1-t)*x_clean + t*x_noise
    x_t = (1.0 - t_target) * x_clean + t_target * x_noise

    # Center of mass removal
    x_t = x_t - x_t.mean(dim=-2, keepdim=True)

    # Convert back to tensor_7 format
    from openfold.utils.rigid_utils import Rotation, Rigid
    rig_t = Rigid(rots=Rotation(rot_mats=R_t), trans=x_t)
    rigids_out = rig_t.to_tensor_7().to(device)

    # Apply flow_mask: only re-noise scaffold, keep motif
    if flow_mask is not None:
        if flow_mask.ndim == 1:
            flow_mask = flow_mask.unsqueeze(0)
        fm = flow_mask.unsqueeze(-1).to(dtype)  # [B, N, 1]
        # motif positions: use motif_rigids if provided, else keep clean
        if motif_rigids is not None:
            if motif_rigids.ndim == 2:
                motif_rigids = motif_rigids.unsqueeze(0)
            motif_src = motif_rigids.to(device).to(dtype)
        else:
            motif_src = rigids_clean.to(dtype)
        rigids_out = fm * rigids_out + (1.0 - fm) * motif_src

    if squeeze:
        rigids_out = rigids_out.squeeze(0)
    return rigids_out


def rigids_to_rot_trans(rigids_t: torch.Tensor):
    """[B, N, 7] quat+trans -> (R [B,N,3,3], x [B,N,3])."""
    from openfold.utils.rigid_utils import Rigid

    rig = Rigid.from_tensor_7(rigids_t)
    R = rig.get_rots().get_rot_mats()
    x = rig.get_trans()
    return R, x
