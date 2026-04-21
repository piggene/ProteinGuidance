"""Iterative refinement experiment for motif scaffolding.

SDEdit/RePaint-style: each round re-noises the previous result to a
decreasing t_start, then re-denoises with guidance. Tests whether
repeated rounds can drive motif RMSD from ~6 Å toward <1 Å.
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
    CA_IDX = 1
    ca = final_atom37[motif_indices, CA_IDX]
    target = target_x.cpu().numpy()
    return float(np.sqrt(((ca - target) ** 2).sum(axis=-1).mean()))


def main():
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "runner", "config")
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="inference")

    cfg.inference.gpu_id = 1
    NUM_T = 50
    cfg.inference.flow.num_t = NUM_T
    LENGTH = 60
    motif_indices_np = np.array([20, 21, 22, 23, 24, 25, 26])
    motif_indices = torch.tensor(motif_indices_np, dtype=torch.long)
    target_R, target_x = make_synthetic_motif(motif_indices_np)
    motif_impute = impute_from_target(LENGTH, motif_indices_np, target_R, target_x)

    print("Building GuidedSampler...")
    sampler = GuidedSampler(cfg)

    Jfn = J_motif(
        target_R=target_R.to(sampler.device),
        target_x=target_x.to(sampler.device),
        motif_indices=motif_indices.to(sampler.device),
        alpha=1.0, beta=1.0, gamma=0.0,
    )

    fixed_mask = np.zeros(LENGTH, dtype=np.float32)
    flow_mask_t = torch.tensor(np.ones(LENGTH, dtype=np.float32))

    # === Iterative refinement schedule ===
    # (t_start, lambda, sigma_rot, sigma_trans, num_t_round)
    schedules = {
        "aggressive": [
            (1.0,  5.0, 0.5, 1.0, 50),
            (0.5,  5.0, 0.5, 1.0, 30),
            (0.3,  8.0, 0.5, 1.0, 20),
            (0.15, 10.0, 0.5, 1.0, 15),
            (0.08, 15.0, 0.5, 1.0, 10),
        ],
        "conservative": [
            (1.0,  5.0, 0.5, 1.0, 50),
            (0.5,  5.0, 0.5, 1.0, 30),
            (0.3,  5.0, 0.5, 1.0, 20),
            (0.2,  5.0, 0.5, 1.0, 15),
            (0.1,  5.0, 0.5, 1.0, 10),
        ],
        "lambda_ramp": [
            (1.0,   2.0, 0.5, 1.0, 50),
            (0.5,   5.0, 0.5, 1.0, 30),
            (0.3,  10.0, 0.5, 1.0, 20),
            (0.15, 20.0, 0.5, 1.0, 15),
            (0.08, 30.0, 0.5, 1.0, 10),
        ],
    }

    all_results = {}

    for sched_name, rounds in schedules.items():
        print(f"\n{'='*60}")
        print(f"Schedule: {sched_name}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        np.random.seed(42)

        rmsds = []
        prev_rigids = None

        for r_idx, (t_start, lam, s_rot, s_trans, nt) in enumerate(rounds):
            print(f"\n--- Round {r_idx+1}: t_start={t_start}, λ={lam}, num_t={nt} ---")

            guide = SimMCGuidance(
                energy_fn=Jfn, K=16, lambda_=lam,
                sigma_rot=s_rot, sigma_trans=s_trans,
            )

            if r_idx == 0:
                out = sampler.sample(
                    sample_length=LENGTH,
                    fixed_mask=fixed_mask,
                    guidance=guide,
                    num_t=nt,
                    verbose=False,
                )
            else:
                # Re-noise previous result
                renoised = renoise_frames(
                    prev_rigids, t_target=t_start,
                    flow_mask=flow_mask_t,
                )
                out = sampler.sample(
                    sample_length=LENGTH,
                    fixed_mask=fixed_mask,
                    guidance=guide,
                    num_t=nt,
                    t_start=t_start,
                    init_rigids=renoised,
                    verbose=False,
                )

            final = out["prot_traj"][0]
            rmsd = motif_ca_rmsd(final, motif_indices_np, target_x)
            rmsds.append(rmsd)
            print(f"  motif RMSD = {rmsd:.3f} Å")

            # Save rigids for next round
            prev_rigids = torch.tensor(out["rigid_traj"][0], dtype=torch.float32)

            glog = out.get("guidance_log", [])
            if glog:
                j_final = glog[-1]["J_mean"]
                print(f"  J_final = {j_final:.1f}")

        all_results[sched_name] = {
            "rounds": rounds,
            "rmsds": rmsds,
        }

    # === Also run single-round baseline for comparison ===
    print(f"\n{'='*60}")
    print("Single-round baseline (no guidance)")
    print(f"{'='*60}")
    torch.manual_seed(42); np.random.seed(42)
    out_bl = sampler.sample(
        sample_length=LENGTH, fixed_mask=fixed_mask,
        guidance=None, num_t=NUM_T, verbose=False,
    )
    rmsd_bl = motif_ca_rmsd(out_bl["prot_traj"][0], motif_indices_np, target_x)
    print(f"  motif RMSD = {rmsd_bl:.3f} Å")

    print(f"\n{'='*60}")
    print("Single-round best (K16 λ=5)")
    print(f"{'='*60}")
    torch.manual_seed(42); np.random.seed(42)
    guide_sr = SimMCGuidance(energy_fn=Jfn, K=16, lambda_=5.0, sigma_rot=0.5, sigma_trans=1.0)
    out_sr = sampler.sample(
        sample_length=LENGTH, fixed_mask=fixed_mask,
        guidance=guide_sr, num_t=NUM_T, verbose=False,
    )
    rmsd_sr = motif_ca_rmsd(out_sr["prot_traj"][0], motif_indices_np, target_x)
    print(f"  motif RMSD = {rmsd_sr:.3f} Å")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: RMSD per round
    ax1 = axes[0]
    colors = {"aggressive": "#E53935", "conservative": "#1E88E5", "lambda_ramp": "#43A047"}
    for sched_name, res in all_results.items():
        rmsds = res["rmsds"]
        ax1.plot(range(1, len(rmsds)+1), rmsds, "o-", color=colors[sched_name],
                 label=sched_name, linewidth=2, markersize=8)
        for i, v in enumerate(rmsds):
            ax1.annotate(f"{v:.1f}", (i+1, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=8)
    ax1.axhline(y=rmsd_bl, color="gray", linestyle="--", alpha=0.7, label=f"baseline ({rmsd_bl:.1f} Å)")
    ax1.axhline(y=rmsd_sr, color="orange", linestyle="--", alpha=0.7, label=f"single-round ({rmsd_sr:.1f} Å)")
    ax1.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="target (1 Å)")
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Motif Cα-RMSD (Å)", fontsize=12)
    ax1.set_title("Iterative Refinement: RMSD per Round")
    ax1.legend(fontsize=9)
    ax1.set_xticks(range(1, 6))

    # Panel 2: comparison bar chart
    ax2 = axes[1]
    bar_labels = ["Baseline", "Single\nround"]
    bar_vals = [rmsd_bl, rmsd_sr]
    bar_colors = ["#999999", "#FF9800"]
    for sched_name, res in all_results.items():
        bar_labels.append(f"{sched_name}\nfinal")
        bar_vals.append(res["rmsds"][-1])
        bar_colors.append(colors[sched_name])
    bars = ax2.bar(range(len(bar_labels)), bar_vals, color=bar_colors,
                   edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(bar_labels)))
    ax2.set_xticklabels(bar_labels, fontsize=9)
    ax2.set_ylabel("Motif Cα-RMSD (Å)", fontsize=12)
    ax2.set_title("Final RMSD Comparison")
    ax2.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="target (1 Å)")
    ax2.legend(fontsize=9)
    for bar, val in zip(bars, bar_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Training-Free Iterative Guidance on FoldFlow++ (motif scaffolding)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "iterative_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline (no guidance):    {rmsd_bl:.3f} Å")
    print(f"Single-round (K16 λ=5):   {rmsd_sr:.3f} Å")
    for sched_name, res in all_results.items():
        rmsds = res["rmsds"]
        print(f"{sched_name:25s}: {' → '.join(f'{r:.1f}' for r in rmsds)} Å")


if __name__ == "__main__":
    main()
