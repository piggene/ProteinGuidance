"""Push toward sub-1 Å motif RMSD.

Strategies:
  A. Extended schedule (12 rounds, t_start down to 0.01)
  B. Higher K (32, 64)
  C. Best-of-16 with extended schedule
  D. Fine-grained final polish (many small steps at low t)
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import compose, initialize_config_dir

from foldflow.guidance import J_motif, SimMCGuidance
from foldflow.guidance.se3n_utils import renoise_frames
from foldflow.guidance.guided_sampler import GuidedSampler
from foldflow.utils.so3_helpers import so3_exp_map

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results_guidance")
os.makedirs(OUT_DIR, exist_ok=True)

CA_IDX = 1


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
    atom_names = {0: " N  ", 1: " CA ", 2: " C  ", 3: " O  "}
    with open(path, "w") as f:
        atom_num = 1
        for res_i in range(atom37.shape[0]):
            for atom_idx, atom_name in atom_names.items():
                x, y, z = atom37[res_i, atom_idx]
                if np.isnan(x) or np.abs(x) > 9999:
                    continue
                b_factor = 90.0 if (motif_indices is not None and res_i in motif_indices) else 50.0
                f.write(
                    f"ATOM  {atom_num:5d} {atom_name} GLY A{res_i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}           "
                    f"{atom_name.strip()[0]:>2s}\n"
                )
                atom_num += 1
        f.write("END\n")


def run_iterative(sampler, Jfn, length, motif_indices_np, target_x,
                  schedule, seed=42, flow_mask_t=None):
    fixed_mask = np.zeros(length, dtype=np.float32)
    if flow_mask_t is None:
        flow_mask_t = torch.tensor(np.ones(length, dtype=np.float32))

    torch.manual_seed(seed)
    np.random.seed(seed)

    rmsds = []
    prev_rigids = None

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

    return rmsds, out


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
    flow_mask_t = torch.tensor(np.ones(LENGTH, dtype=np.float32))

    print("Building GuidedSampler...")
    sampler = GuidedSampler(cfg)

    Jfn = J_motif(
        target_R=target_R.to(sampler.device),
        target_x=target_x.to(sampler.device),
        motif_indices=motif_indices.to(sampler.device),
        alpha=1.0, beta=1.0, gamma=0.0,
    )

    # === Schedule A: 12-round extended, K=16 ===
    sched_A = [
        (1.0,  5.0, 16, 50),
        (0.5,  5.0, 16, 30),
        (0.3,  5.0, 16, 20),
        (0.2,  5.0, 16, 15),
        (0.1,  5.0, 16, 10),
        (0.07, 5.0, 16, 10),
        (0.05, 5.0, 16, 8),
        (0.03, 5.0, 16, 8),
        (0.02, 5.0, 16, 6),
        (0.015,5.0, 16, 6),
        (0.01, 5.0, 16, 6),
        (0.01, 5.0, 16, 6),
    ]

    # === Schedule B: 12-round, K=32 ===
    sched_B = [
        (1.0,  5.0, 32, 50),
        (0.5,  5.0, 32, 30),
        (0.3,  5.0, 32, 20),
        (0.2,  5.0, 32, 15),
        (0.1,  5.0, 32, 10),
        (0.07, 5.0, 32, 10),
        (0.05, 5.0, 32, 8),
        (0.03, 5.0, 32, 8),
        (0.02, 5.0, 32, 6),
        (0.015,5.0, 32, 6),
        (0.01, 5.0, 32, 6),
        (0.01, 5.0, 32, 6),
    ]

    # === Schedule C: 12-round, K=64 last 4 rounds ===
    sched_C = [
        (1.0,  5.0, 16, 50),
        (0.5,  5.0, 16, 30),
        (0.3,  5.0, 16, 20),
        (0.2,  5.0, 16, 15),
        (0.1,  5.0, 32, 10),
        (0.07, 5.0, 32, 10),
        (0.05, 5.0, 32, 8),
        (0.03, 5.0, 32, 8),
        (0.02, 5.0, 64, 8),
        (0.015,5.0, 64, 8),
        (0.01, 5.0, 64, 8),
        (0.01, 5.0, 64, 8),
    ]

    # === Schedule D: fine-grained polish (more steps at small t) ===
    sched_D = [
        (1.0,  5.0, 16, 50),
        (0.5,  5.0, 16, 30),
        (0.3,  5.0, 16, 20),
        (0.2,  5.0, 16, 15),
        (0.1,  5.0, 16, 15),
        (0.07, 5.0, 32, 15),
        (0.05, 5.0, 32, 15),
        (0.03, 5.0, 32, 15),
        (0.02, 5.0, 32, 15),
        (0.015,5.0, 32, 15),
        (0.01, 5.0, 32, 15),
        (0.01, 5.0, 32, 15),
    ]

    schedules = {
        "A_K16_12r": sched_A,
        "B_K32_12r": sched_B,
        "C_K64_tail": sched_C,
        "D_fine_steps": sched_D,
    }

    all_results = {}

    # Run each schedule with seed=42
    for name, sched in schedules.items():
        print(f"\n{'='*60}")
        print(f"Schedule {name}")
        print("="*60)
        rmsds, out = run_iterative(
            sampler, Jfn, LENGTH, motif_indices_np, target_x,
            sched, seed=42, flow_mask_t=flow_mask_t,
        )
        all_results[name] = {"rmsds": rmsds, "out": out}
        for i, r in enumerate(rmsds):
            marker = " ***" if r < 1.0 else (" *" if r < 1.5 else "")
            print(f"  Round {i+1:2d} (t={sched[i][0]:.3f}, K={sched[i][2]:2d}): {r:.3f} Å{marker}")

    # Best-of-16 with best schedule
    best_sched_name = min(all_results, key=lambda k: all_results[k]["rmsds"][-1])
    best_sched = schedules[best_sched_name]
    print(f"\n{'='*60}")
    print(f"Best-of-16 with schedule {best_sched_name}")
    print("="*60)

    N_SAMPLES = 16
    all_final_rmsds = []
    best_rmsd = float("inf")
    best_out = None
    best_seed = None
    best_trace = None

    for s in range(N_SAMPLES):
        seed = 200 + s
        rmsds_s, out_s = run_iterative(
            sampler, Jfn, LENGTH, motif_indices_np, target_x,
            best_sched, seed=seed, flow_mask_t=flow_mask_t,
        )
        final_rmsd = rmsds_s[-1]
        all_final_rmsds.append(final_rmsd)
        marker = " ***" if final_rmsd < 1.0 else (" *" if final_rmsd < 1.5 else "")
        print(f"  Sample {s+1:2d} (seed={seed}): final={final_rmsd:.3f} Å{marker}")
        if final_rmsd < best_rmsd:
            best_rmsd = final_rmsd
            best_out = out_s
            best_seed = seed
            best_trace = rmsds_s

    print(f"\n  Best: seed={best_seed}, RMSD={best_rmsd:.3f} Å")
    sorted_rmsds = sorted(all_final_rmsds)
    print(f"  All (sorted): {[f'{r:.2f}' for r in sorted_rmsds]}")
    print(f"  Mean: {np.mean(all_final_rmsds):.3f} ± {np.std(all_final_rmsds):.3f} Å")
    under_1 = sum(1 for r in all_final_rmsds if r < 1.0)
    under_1_5 = sum(1 for r in all_final_rmsds if r < 1.5)
    under_2 = sum(1 for r in all_final_rmsds if r < 2.0)
    under_3 = sum(1 for r in all_final_rmsds if r < 3.0)
    print(f"  < 1.0 Å: {under_1}/{N_SAMPLES}  |  < 1.5 Å: {under_1_5}/{N_SAMPLES}  |  < 2.0 Å: {under_2}/{N_SAMPLES}  |  < 3.0 Å: {under_3}/{N_SAMPLES}")

    # Save best PDB
    if best_out is not None:
        final_best = best_out["prot_traj"][0]
        write_pdb(final_best, os.path.join(OUT_DIR, "push_best.pdb"), motif_indices_np)

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Schedule comparison
    ax1 = axes[0]
    colors_sched = {"A_K16_12r": "#1E88E5", "B_K32_12r": "#E53935",
                    "C_K64_tail": "#43A047", "D_fine_steps": "#FF9800"}
    for name, res in all_results.items():
        rmsds = res["rmsds"]
        ax1.plot(range(1, len(rmsds)+1), rmsds, "o-", color=colors_sched[name],
                 linewidth=2, markersize=6, label=f"{name} (final={rmsds[-1]:.2f})")
    ax1.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="target (1 Å)")
    ax1.set_xlabel("Round"); ax1.set_ylabel("Motif Cα-RMSD (Å)")
    ax1.set_title("Schedule Comparison (12 rounds)")
    ax1.legend(fontsize=8); ax1.set_xticks(range(1, 13))

    # Panel 2: Best-of-16 histogram
    ax2 = axes[1]
    ax2.hist(all_final_rmsds, bins=12, color="#FF9800", edgecolor="black", alpha=0.8)
    ax2.axvline(x=1.0, color="red", linestyle=":", linewidth=2, label="target (1 Å)")
    ax2.axvline(x=best_rmsd, color="#E53935", linestyle="--", linewidth=2,
                label=f"best ({best_rmsd:.2f} Å)")
    ax2.axvline(x=np.mean(all_final_rmsds), color="#1E88E5", linestyle="--", linewidth=2,
                label=f"mean ({np.mean(all_final_rmsds):.2f} Å)")
    ax2.set_xlabel("Final motif RMSD (Å)"); ax2.set_ylabel("Count")
    ax2.set_title(f"Best-of-{N_SAMPLES} Distribution ({best_sched_name})")
    ax2.legend(fontsize=9)

    # Panel 3: Best sample convergence
    ax3 = axes[2]
    if best_trace:
        ax3.plot(range(1, len(best_trace)+1), best_trace, "o-", color="#E53935",
                 linewidth=2, markersize=8, label=f"best (seed={best_seed})")
        for i, v in enumerate(best_trace):
            ax3.annotate(f"{v:.2f}", (i+1, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=8)
    ax3.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="target (1 Å)")
    ax3.set_xlabel("Round"); ax3.set_ylabel("Motif Cα-RMSD (Å)")
    ax3.set_title("Best Sample Convergence"); ax3.legend(fontsize=9)
    ax3.set_xticks(range(1, 13))

    fig.suptitle("Pushing Toward Sub-1Å: Extended Iterative Guidance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "push_limit.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    for name, res in all_results.items():
        print(f"  {name:20s}: {res['rmsds'][-1]:.3f} Å (final)")
    print(f"  {'Best-of-16':20s}: {best_rmsd:.3f} Å (seed={best_seed})")
    print(f"  {'Mean±std':20s}: {np.mean(all_final_rmsds):.3f} ± {np.std(all_final_rmsds):.3f} Å")
    print(f"  Success < 1 Å:      {under_1}/{N_SAMPLES}")
    print(f"  Near    < 1.5 Å:    {under_1_5}/{N_SAMPLES}")
    print(f"  Viable  < 2 Å:      {under_2}/{N_SAMPLES}")


if __name__ == "__main__":
    main()
