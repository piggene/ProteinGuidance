"""Guidance sanity checks on FoldFlow++.

Two tests:
  A. Plant motif via fixed_mask. Verify the motif positions at the final sample
     match the imputed target (motif_RMSD ~ 0). Tests masking plumbing.
  B. No mask (fixed_mask=0 everywhere). Define J_motif at the same residues and
     run SimMCGuidance. Verify motif_RMSD drops meaningfully vs. unguided.
     Tests the guidance path actually steers the flow.

Keeps num_t, K, length small to minimize GPU time.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import compose, initialize_config_dir

from foldflow.guidance import J_motif, SimMCGuidance
from foldflow.guidance.guided_sampler import GuidedSampler
from foldflow.utils.so3_helpers import so3_exp_map


def make_synthetic_motif(motif_indices, seed=42, spacing=3.8):
    """Fake target: identity rotations and equally spaced Cα positions.

    Returns target_R [M,3,3], target_x [M,3] (in Angstroms).
    """
    rng = np.random.default_rng(seed)
    M = len(motif_indices)
    # Slight perturbation of identity for the target rotations.
    rotvec = rng.standard_normal((M, 3)) * 0.3
    R = so3_exp_map(torch.tensor(rotvec, dtype=torch.float64)).numpy()
    # Positions: line along x axis with small jitter.
    x = np.stack([spacing * np.arange(M) + rng.normal(0, 0.1, M),
                  rng.normal(0, 0.1, M),
                  rng.normal(0, 0.1, M)], axis=-1)
    # Center so COM=0 (FoldFlow++ centers translations).
    x = x - x.mean(axis=0, keepdims=True)
    return torch.tensor(R, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)


def rigids_tensor_7_from_RT(R, x):
    """[M,3,3] and [M,3] -> [M,7] (quat_wxyz + trans)."""
    from openfold.utils.rigid_utils import Rotation, Rigid
    rots = Rotation(rot_mats=R.to(torch.float32))
    trans = x.to(torch.float32)
    rig = Rigid(rots=rots, trans=trans)
    return rig.to_tensor_7()


def impute_from_target(length, motif_indices, target_R, target_x):
    """Build [N,7] with target frames at motif positions, zeros elsewhere."""
    from openfold.utils.rigid_utils import Rotation, Rigid
    N = length
    R_full = torch.eye(3).unsqueeze(0).expand(N, 3, 3).clone()
    x_full = torch.zeros(N, 3)
    R_full[motif_indices] = target_R
    x_full[motif_indices] = target_x
    return rigids_tensor_7_from_RT(R_full, x_full)


def motif_ca_rmsd(final_atom37, motif_indices, target_x):
    """Cα RMSD at motif positions between final sample and target.

    `final_atom37` [N, 37, 3]. Cα is atom index 1 in atom37.
    """
    CA_IDX = 1
    ca = final_atom37[motif_indices, CA_IDX]  # [M, 3]
    target = target_x.cpu().numpy()
    # Align via translation only (rotation-free RMSD is fine; comparing raw positions
    # in FoldFlow++'s centered frame).
    return float(np.sqrt(((ca - target) ** 2).sum(axis=-1).mean()))


def run_once(sampler, length, motif_indices, target_R, target_x,
             fixed_mask, guidance, num_t, label):
    impute = impute_from_target(length, motif_indices, target_R, target_x).to(sampler.device)
    out = sampler.sample(
        sample_length=length,
        fixed_mask=fixed_mask,
        motif_rigids_impute=impute if fixed_mask.sum() > 0 else None,
        guidance=guidance,
        num_t=num_t,
        verbose=False,
    )
    final = out["prot_traj"][0]  # atom37 at t=eps (end of reverse)
    rmsd = motif_ca_rmsd(final, motif_indices, target_x)
    print(f"[{label}] motif Cα-RMSD = {rmsd:.3f} Å")
    return rmsd, out


def main():
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "runner", "config")
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="inference")

    cfg.inference.gpu_id = 1
    cfg.inference.flow.num_t = 15

    length = 40
    motif_indices_np = np.array([15, 16, 17, 18, 19])
    motif_indices = torch.tensor(motif_indices_np, dtype=torch.long)
    target_R, target_x = make_synthetic_motif(motif_indices_np)

    print("Building GuidedSampler...")
    sampler = GuidedSampler(cfg)

    # --- Test A: planted motif via fixed_mask ---
    fixed_mask_A = np.zeros(length, dtype=np.float32)
    fixed_mask_A[motif_indices_np] = 1.0
    run_once(sampler, length, motif_indices_np, target_R, target_x,
             fixed_mask_A, guidance=None, num_t=15, label="A planted-mask")

    # --- Test B: guidance only (no fixed_mask), compare to unguided ---
    fixed_mask_B = np.zeros(length, dtype=np.float32)

    Jfn = J_motif(
        target_R=target_R.to(sampler.device),
        target_x=target_x.to(sampler.device),
        motif_indices=motif_indices.to(sampler.device),
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
    )

    print("\n--- Unguided (reference, num_t=30) ---")
    torch.manual_seed(123); np.random.seed(123)
    rmsd_unguided, _ = run_once(
        sampler, length, motif_indices_np, target_R, target_x,
        fixed_mask_B, guidance=None, num_t=30, label="B unguided"
    )

    for K, lam in [(8, 2.0), (8, 5.0), (16, 2.0), (16, 5.0)]:
        guide = SimMCGuidance(
            energy_fn=Jfn, K=K, lambda_=lam,
            sigma_rot=0.3, sigma_trans=0.5,
        )
        torch.manual_seed(123); np.random.seed(123)
        rmsd_guided, out = run_once(
            sampler, length, motif_indices_np, target_R, target_x,
            fixed_mask_B, guidance=guide, num_t=30,
            label=f"B sim-MC K={K} λ={lam}"
        )
        log = out.get("guidance_log", [])
        if log:
            j_mean = np.mean([r["J_mean"] for r in log])
            g_r = np.mean([r["g_rot_norm"] for r in log])
            g_x = np.mean([r["g_trans_norm"] for r in log])
            print(f"    avg J={j_mean:.2f}  |g_rot|={g_r:.3f}  |g_trans|={g_x:.3f}"
                  f"  ratio={rmsd_guided/rmsd_unguided:.3f}")


if __name__ == "__main__":
    main()
