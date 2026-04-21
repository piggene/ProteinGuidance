"""Full evaluation: extended iterative refinement + best-of-N + structure visualization.

Tests whether training-free guidance can reach motif RMSD < 1 Å.
Saves PDB files and backbone visualizations.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import compose, initialize_config_dir

from foldflow.guidance import J_motif, SimMCGuidance
from foldflow.guidance.se3n_utils import renoise_frames
from foldflow.guidance.guided_sampler import GuidedSampler
from foldflow.utils.so3_helpers import so3_exp_map

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results_guidance")
os.makedirs(OUT_DIR, exist_ok=True)

CA_IDX = 1
N_IDX = 0
C_IDX = 2
O_IDX = 3


def make_synthetic_motif(motif_indices, seed=42, spacing=3.8):
    rng = np.random.default_rng(seed)
    M = len(motif_indices)
    rotvec = rng.standard_normal((M, 3)) * 0.3
    R = so3_exp_map(torch.tensor(rotvec, dtype=torch.float64)).numpy()
    x = np.stack([spacing * np.arange(M) + rng.normal(0, 0.1, M),
                  rng.normal(0, 0.1, M),
                  rng.normal(0, 0.1, M)], axis=-1)
    x = x - x.mean(axis=0, keepdims=True)
    return torch.tensor(R, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)


def rigids_tensor_7_from_RT(R, x):
    from openfold.utils.rigid_utils import Rotation, Rigid
    rots = Rotation(rot_mats=R.to(torch.float32))
    rig = Rigid(rots=rots, trans=x.to(torch.float32))
    return rig.to_tensor_7()


def impute_from_target(length, motif_indices, target_R, target_x):
    N = length
    R_full = torch.eye(3).unsqueeze(0).expand(N, 3, 3).clone()
    x_full = torch.zeros(N, 3)
    R_full[motif_indices] = target_R
    x_full[motif_indices] = target_x
    return rigids_tensor_7_from_RT(R_full, x_full)


def motif_ca_rmsd(final_atom37, motif_indices, target_x):
    ca = final_atom37[motif_indices, CA_IDX]
    target = target_x.cpu().numpy()
    return float(np.sqrt(((ca - target) ** 2).sum(axis=-1).mean()))


def write_pdb(atom37, path, motif_indices=None):
    """Write atom37 [N, 37, 3] to PDB with backbone atoms."""
    atom_names = {N_IDX: " N  ", CA_IDX: " CA ", C_IDX: " C  ", O_IDX: " O  "}
    with open(path, "w") as f:
        atom_num = 1
        for res_i in range(atom37.shape[0]):
            chain = "A"
            resname = "GLY"
            for atom_idx, atom_name in atom_names.items():
                x, y, z = atom37[res_i, atom_idx]
                if np.isnan(x) or np.abs(x) > 9999:
                    continue
                b_factor = 50.0
                if motif_indices is not None and res_i in motif_indices:
                    b_factor = 90.0
                f.write(
                    f"ATOM  {atom_num:5d} {atom_name} {resname} {chain}{res_i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}           "
                    f"{atom_name.strip()[0]:>2s}\n"
                )
                atom_num += 1
        f.write("END\n")


def plot_backbone(atom37, motif_indices, target_x, title, path):
    """3-panel backbone visualization: 3D + two 2D projections."""
    ca = atom37[:, CA_IDX]
    N = len(ca)
    motif_set = set(motif_indices)
    scaffold_idx = [i for i in range(N) if i not in motif_set]

    fig = plt.figure(figsize=(18, 6))

    # 3D view
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(ca[scaffold_idx, 0], ca[scaffold_idx, 1], ca[scaffold_idx, 2],
             "o-", color="#2196F3", markersize=3, linewidth=1, alpha=0.6, label="scaffold")
    ax1.plot(ca[motif_indices, 0], ca[motif_indices, 1], ca[motif_indices, 2],
             "o-", color="#E53935", markersize=6, linewidth=2, label="generated motif")
    target_np = target_x.cpu().numpy()
    ax1.plot(target_np[:, 0], target_np[:, 1], target_np[:, 2],
             "x-", color="#4CAF50", markersize=10, linewidth=2, label="target motif")
    # Chain trace
    ax1.plot(ca[:, 0], ca[:, 1], ca[:, 2], "-", color="gray", linewidth=0.5, alpha=0.3)
    ax1.set_title("3D backbone", fontsize=11)
    ax1.legend(fontsize=8)

    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.plot(ca[:, 0], ca[:, 1], "-", color="gray", linewidth=0.5, alpha=0.3)
    ax2.scatter(ca[scaffold_idx, 0], ca[scaffold_idx, 1], c="#2196F3", s=15, alpha=0.6, label="scaffold")
    ax2.scatter(ca[motif_indices, 0], ca[motif_indices, 1], c="#E53935", s=50, zorder=5, label="generated motif")
    ax2.scatter(target_np[:, 0], target_np[:, 1], c="#4CAF50", s=80, marker="x", zorder=6, label="target motif")
    for i, mi in enumerate(motif_indices):
        ax2.annotate("", xy=(ca[mi, 0], ca[mi, 1]), xytext=(target_np[i, 0], target_np[i, 1]),
                     arrowprops=dict(arrowstyle="->", color="orange", lw=1.5, alpha=0.7))
    ax2.set_xlabel("X (Å)"); ax2.set_ylabel("Y (Å)")
    ax2.set_title("XY projection", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")

    # XZ projection
    ax3 = fig.add_subplot(133)
    ax3.plot(ca[:, 0], ca[:, 2], "-", color="gray", linewidth=0.5, alpha=0.3)
    ax3.scatter(ca[scaffold_idx, 0], ca[scaffold_idx, 2], c="#2196F3", s=15, alpha=0.6, label="scaffold")
    ax3.scatter(ca[motif_indices, 0], ca[motif_indices, 2], c="#E53935", s=50, zorder=5, label="generated motif")
    ax3.scatter(target_np[:, 0], target_np[:, 2], c="#4CAF50", s=80, marker="x", zorder=6, label="target motif")
    for i, mi in enumerate(motif_indices):
        ax3.annotate("", xy=(ca[mi, 0], ca[mi, 2]), xytext=(target_np[i, 0], target_np[i, 2]),
                     arrowprops=dict(arrowstyle="->", color="orange", lw=1.5, alpha=0.7))
    ax3.set_xlabel("X (Å)"); ax3.set_ylabel("Z (Å)")
    ax3.set_title("XZ projection", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.set_aspect("equal")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_iterative(sampler, Jfn, length, motif_indices_np, target_R, target_x,
                  schedule, seed=42):
    """Run one full iterative refinement and return per-round info."""
    fixed_mask = np.zeros(length, dtype=np.float32)
    flow_mask_t = torch.tensor(np.ones(length, dtype=np.float32))

    torch.manual_seed(seed)
    np.random.seed(seed)

    rmsds = []
    prev_rigids = None
    final_out = None

    for r_idx, (t_start, lam, K, nt) in enumerate(schedule):
        guide = SimMCGuidance(
            energy_fn=Jfn, K=K, lambda_=lam,
            sigma_rot=0.5, sigma_trans=1.0,
        )
        if r_idx == 0:
            out = sampler.sample(
                sample_length=length, fixed_mask=fixed_mask,
                guidance=guide, num_t=nt, verbose=False,
            )
        else:
            renoised = renoise_frames(prev_rigids, t_target=t_start, flow_mask=flow_mask_t)
            out = sampler.sample(
                sample_length=length, fixed_mask=fixed_mask,
                guidance=guide, num_t=nt, t_start=t_start,
                init_rigids=renoised, verbose=False,
            )

        final = out["prot_traj"][0]
        rmsd = motif_ca_rmsd(final, motif_indices_np, target_x)
        rmsds.append(rmsd)
        prev_rigids = torch.tensor(out["rigid_traj"][0], dtype=torch.float32)
        final_out = out

    return rmsds, final_out


def main():
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "runner", "config")
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="inference")

    cfg.inference.gpu_id = 1
    LENGTH = 60
    motif_indices_np = np.array([20, 21, 22, 23, 24, 25, 26])
    motif_indices = torch.tensor(motif_indices_np, dtype=torch.long)
    target_R, target_x = make_synthetic_motif(motif_indices_np)

    print("Building GuidedSampler...")
    sampler = GuidedSampler(cfg)

    Jfn = J_motif(
        target_R=target_R.to(sampler.device),
        target_x=target_x.to(sampler.device),
        motif_indices=motif_indices.to(sampler.device),
        alpha=1.0, beta=1.0, gamma=0.0,
    )

    # Extended conservative schedule: 8 rounds, going to t_start=0.03
    # (t_start, lambda, K, num_t)
    extended_schedule = [
        (1.0,  5.0, 16, 50),
        (0.5,  5.0, 16, 30),
        (0.3,  5.0, 16, 20),
        (0.2,  5.0, 16, 15),
        (0.1,  5.0, 16, 10),
        (0.07, 5.0, 32, 10),
        (0.05, 5.0, 32, 8),
        (0.03, 5.0, 32, 6),
    ]

    # ============================================================
    # Experiment 1: Extended iterative (single sample)
    # ============================================================
    print("\n" + "="*60)
    print("Exp 1: Extended iterative refinement (8 rounds)")
    print("="*60)
    rmsds_ext, out_ext = run_iterative(
        sampler, Jfn, LENGTH, motif_indices_np, target_R, target_x,
        extended_schedule, seed=42,
    )
    for i, r in enumerate(rmsds_ext):
        t_s = extended_schedule[i][0]
        print(f"  Round {i+1} (t={t_s}): {r:.3f} Å")

    # Save structure
    final_ext = out_ext["prot_traj"][0]
    write_pdb(final_ext, os.path.join(OUT_DIR, "extended_final.pdb"), motif_indices_np)
    plot_backbone(final_ext, motif_indices_np, target_x,
                  f"Extended iterative (8 rounds) — motif RMSD = {rmsds_ext[-1]:.2f} Å",
                  os.path.join(OUT_DIR, "extended_backbone.png"))

    # ============================================================
    # Experiment 2: Best-of-N with iterative refinement
    # ============================================================
    N_SAMPLES = 8
    print(f"\n{'='*60}")
    print(f"Exp 2: Best-of-{N_SAMPLES} iterative refinement")
    print("="*60)

    best_rmsd = float("inf")
    best_out = None
    best_seed = None
    all_final_rmsds = []

    for s in range(N_SAMPLES):
        seed = 100 + s
        rmsds_s, out_s = run_iterative(
            sampler, Jfn, LENGTH, motif_indices_np, target_R, target_x,
            extended_schedule, seed=seed,
        )
        final_rmsd = rmsds_s[-1]
        all_final_rmsds.append(final_rmsd)
        print(f"  Sample {s+1} (seed={seed}): {' → '.join(f'{r:.1f}' for r in rmsds_s)} Å")
        if final_rmsd < best_rmsd:
            best_rmsd = final_rmsd
            best_out = out_s
            best_seed = seed
            best_rmsds_trace = rmsds_s

    print(f"\n  Best: seed={best_seed}, RMSD={best_rmsd:.3f} Å")
    print(f"  All final RMSDs: {[f'{r:.2f}' for r in sorted(all_final_rmsds)]}")
    print(f"  Mean: {np.mean(all_final_rmsds):.2f} ± {np.std(all_final_rmsds):.2f} Å")
    under_2 = sum(1 for r in all_final_rmsds if r < 2.0)
    under_3 = sum(1 for r in all_final_rmsds if r < 3.0)
    under_5 = sum(1 for r in all_final_rmsds if r < 5.0)
    print(f"  < 2 Å: {under_2}/{N_SAMPLES}  |  < 3 Å: {under_3}/{N_SAMPLES}  |  < 5 Å: {under_5}/{N_SAMPLES}")

    # Save best structure
    final_best = best_out["prot_traj"][0]
    write_pdb(final_best, os.path.join(OUT_DIR, "best_of_n.pdb"), motif_indices_np)
    plot_backbone(final_best, motif_indices_np, target_x,
                  f"Best-of-{N_SAMPLES} — motif RMSD = {best_rmsd:.2f} Å (seed={best_seed})",
                  os.path.join(OUT_DIR, "best_backbone.png"))

    # ============================================================
    # Experiment 3: Baseline (no guidance) for comparison
    # ============================================================
    print(f"\n{'='*60}")
    print("Exp 3: Baseline (no guidance)")
    print("="*60)
    torch.manual_seed(42); np.random.seed(42)
    out_bl = sampler.sample(
        sample_length=LENGTH, fixed_mask=np.zeros(LENGTH, dtype=np.float32),
        guidance=None, num_t=50, verbose=False,
    )
    rmsd_bl = motif_ca_rmsd(out_bl["prot_traj"][0], motif_indices_np, target_x)
    print(f"  motif RMSD = {rmsd_bl:.3f} Å")
    final_bl = out_bl["prot_traj"][0]
    write_pdb(final_bl, os.path.join(OUT_DIR, "baseline.pdb"), motif_indices_np)
    plot_backbone(final_bl, motif_indices_np, target_x,
                  f"Baseline (no guidance) — motif RMSD = {rmsd_bl:.2f} Å",
                  os.path.join(OUT_DIR, "baseline_backbone.png"))

    # ============================================================
    # Summary figure
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: RMSD convergence
    ax1 = axes[0]
    ax1.plot(range(1, len(rmsds_ext)+1), rmsds_ext, "o-", color="#1E88E5",
             linewidth=2, markersize=8, label="single sample (seed=42)")
    ax1.plot(range(1, len(best_rmsds_trace)+1), best_rmsds_trace, "s--", color="#E53935",
             linewidth=2, markersize=8, label=f"best-of-{N_SAMPLES} (seed={best_seed})")
    for i, v in enumerate(rmsds_ext):
        ax1.annotate(f"{v:.1f}", (i+1, v), textcoords="offset points", xytext=(8, 5), fontsize=8, color="#1E88E5")
    for i, v in enumerate(best_rmsds_trace):
        ax1.annotate(f"{v:.1f}", (i+1, v), textcoords="offset points", xytext=(8, -12), fontsize=8, color="#E53935")
    ax1.axhline(y=rmsd_bl, color="gray", linestyle="--", alpha=0.5, label=f"baseline ({rmsd_bl:.1f} Å)")
    ax1.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="target (1 Å)")
    ax1.set_xlabel("Round"); ax1.set_ylabel("Motif Cα-RMSD (Å)")
    ax1.set_title("RMSD Convergence"); ax1.legend(fontsize=8)
    ax1.set_xticks(range(1, 9))

    # Panel 2: best-of-N histogram
    ax2 = axes[1]
    ax2.hist(all_final_rmsds, bins=10, color="#FF9800", edgecolor="black", alpha=0.8)
    ax2.axvline(x=1.0, color="red", linestyle=":", label="target (1 Å)")
    ax2.axvline(x=best_rmsd, color="#E53935", linestyle="--", label=f"best ({best_rmsd:.2f} Å)")
    ax2.axvline(x=np.mean(all_final_rmsds), color="#1E88E5", linestyle="--",
                label=f"mean ({np.mean(all_final_rmsds):.2f} Å)")
    ax2.set_xlabel("Final motif RMSD (Å)"); ax2.set_ylabel("Count")
    ax2.set_title(f"Best-of-{N_SAMPLES} Distribution"); ax2.legend(fontsize=8)

    # Panel 3: comparison bar
    ax3 = axes[2]
    labels = ["Baseline", "Single\n(1 round)", "Iterative\n(8 rounds)", f"Best-of-{N_SAMPLES}\n(8 rounds)"]
    vals = [rmsd_bl, rmsds_ext[0], rmsds_ext[-1], best_rmsd]
    colors = ["#999999", "#FF9800", "#1E88E5", "#E53935"]
    bars = ax3.bar(range(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(range(len(labels))); ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel("Motif Cα-RMSD (Å)")
    ax3.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="target (1 Å)")
    ax3.set_title("Method Comparison"); ax3.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Training-Free Iterative Guidance: Full Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "full_eval.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*60}")
    print("FILES SAVED")
    print("="*60)
    for f in ["full_eval.png", "extended_backbone.png", "best_backbone.png",
              "baseline_backbone.png", "extended_final.pdb", "best_of_n.pdb", "baseline.pdb"]:
        p = os.path.join(OUT_DIR, f)
        print(f"  {p}" + (" ✓" if os.path.exists(p) else " ✗"))

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  Baseline:              {rmsd_bl:.2f} Å")
    print(f"  Single-round guided:   {rmsds_ext[0]:.2f} Å")
    print(f"  8-round iterative:     {rmsds_ext[-1]:.2f} Å")
    print(f"  Best-of-{N_SAMPLES} iterative:  {best_rmsd:.2f} Å")
    print(f"  Target:                < 1.00 Å")
    print(f"  Success (< 1 Å):      {sum(1 for r in all_final_rmsds if r < 1.0)}/{N_SAMPLES}")
    print(f"  Near-miss (< 2 Å):    {under_2}/{N_SAMPLES}")
    print(f"  Viable (< 3 Å):       {under_3}/{N_SAMPLES}")


if __name__ == "__main__":
    main()
