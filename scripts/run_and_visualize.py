"""Run guidance experiments and produce visualization plots.

Runs:
  1. Baseline (no guidance, no mask)
  2. Planted motif (fixed_mask only, no guidance) — sanity check
  3. SimMCGuidance sweep over (K, lambda, sigma_trans)

Saves:
  - results.npz with all RMSD values and guidance logs
  - guidance_results.png — bar chart of motif RMSD + per-step J/g_norm curves
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import compose, initialize_config_dir

from foldflow.guidance import J_motif, SimMCGuidance
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


def run_once(sampler, length, motif_indices, target_R, target_x,
             fixed_mask, guidance, num_t, seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    impute = impute_from_target(length, motif_indices, target_R, target_x).to(sampler.device)
    out = sampler.sample(
        sample_length=length,
        fixed_mask=fixed_mask,
        motif_rigids_impute=impute if fixed_mask.sum() > 0 else None,
        guidance=guidance,
        num_t=num_t,
        verbose=True,
    )
    final = out["prot_traj"][0]
    rmsd = motif_ca_rmsd(final, motif_indices, target_x)
    return rmsd, out


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

    print("Building GuidedSampler...")
    sampler = GuidedSampler(cfg)

    Jfn = J_motif(
        target_R=target_R.to(sampler.device),
        target_x=target_x.to(sampler.device),
        motif_indices=motif_indices.to(sampler.device),
        alpha=1.0, beta=1.0, gamma=0.0,
    )

    results = {}

    # 1) Baseline: no guidance, no mask
    print("\n=== Baseline (unguided) ===")
    rmsd_bl, out_bl = run_once(sampler, LENGTH, motif_indices_np, target_R, target_x,
                               np.zeros(LENGTH, dtype=np.float32), None, NUM_T)
    results["baseline"] = {"rmsd": rmsd_bl}
    print(f"  motif RMSD = {rmsd_bl:.3f} Å")

    # 2) Planted mask
    print("\n=== Planted motif (fixed_mask) ===")
    fm_planted = np.zeros(LENGTH, dtype=np.float32)
    fm_planted[motif_indices_np] = 1.0
    rmsd_pl, out_pl = run_once(sampler, LENGTH, motif_indices_np, target_R, target_x,
                               fm_planted, None, NUM_T)
    results["planted"] = {"rmsd": rmsd_pl}
    print(f"  motif RMSD = {rmsd_pl:.3f} Å (should be ~0)")

    # 3) SimMC sweep
    sweep_configs = [
        {"K": 16, "lambda_": 2.0,  "sigma_rot": 0.5, "sigma_trans": 1.0, "label": "K16_λ2_σ0.5"},
        {"K": 16, "lambda_": 5.0,  "sigma_rot": 0.5, "sigma_trans": 1.0, "label": "K16_λ5_σ0.5"},
        {"K": 16, "lambda_": 10.0, "sigma_rot": 0.5, "sigma_trans": 1.0, "label": "K16_λ10_σ0.5"},
        {"K": 16, "lambda_": 5.0,  "sigma_rot": 1.0, "sigma_trans": 2.0, "label": "K16_λ5_σ1.0"},
        {"K": 16, "lambda_": 10.0, "sigma_rot": 1.0, "sigma_trans": 2.0, "label": "K16_λ10_σ1.0"},
        {"K": 32, "lambda_": 5.0,  "sigma_rot": 0.5, "sigma_trans": 1.0, "label": "K32_λ5_σ0.5"},
        {"K": 32, "lambda_": 10.0, "sigma_rot": 1.0, "sigma_trans": 2.0, "label": "K32_λ10_σ1.0"},
    ]

    fm_none = np.zeros(LENGTH, dtype=np.float32)
    for sc in sweep_configs:
        label = sc["label"]
        print(f"\n=== SimMC {label} ===")
        guide = SimMCGuidance(
            energy_fn=Jfn, K=sc["K"], lambda_=sc["lambda_"],
            sigma_rot=sc["sigma_rot"], sigma_trans=sc["sigma_trans"],
        )
        rmsd_g, out_g = run_once(sampler, LENGTH, motif_indices_np, target_R, target_x,
                                 fm_none, guide, NUM_T)
        glog = out_g.get("guidance_log", [])
        results[label] = {
            "rmsd": rmsd_g,
            "guidance_log": glog,
            "config": {k: v for k, v in sc.items() if k != "label"},
        }
        print(f"  motif RMSD = {rmsd_g:.3f} Å  (ratio vs baseline = {rmsd_g/rmsd_bl:.3f})")

    # === Save ===
    np.savez(os.path.join(OUT_DIR, "results.npz"), results=results)
    print(f"\nResults saved to {OUT_DIR}/results.npz")

    # === Plot ===
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: RMSD bar chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    labels_bar = ["Baseline\n(unguided)", "Planted\n(mask)"]
    rmsds_bar = [results["baseline"]["rmsd"], results["planted"]["rmsd"]]
    colors_bar = ["#999999", "#4CAF50"]
    for sc in sweep_configs:
        labels_bar.append(sc["label"].replace("_", "\n"))
        rmsds_bar.append(results[sc["label"]]["rmsd"])
        colors_bar.append("#2196F3")
    bars = ax1.bar(range(len(labels_bar)), rmsds_bar, color=colors_bar, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(labels_bar)))
    ax1.set_xticklabels(labels_bar, fontsize=8)
    ax1.set_ylabel("Motif Cα-RMSD (Å)")
    ax1.set_title("Motif RMSD: Baseline vs Guidance")
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1 Å target")
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, rmsds_bar):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # --- Panel 2: RMSD ratio vs baseline ---
    ax2 = fig.add_subplot(gs[0, 1])
    guided_labels = []
    guided_ratios = []
    for sc in sweep_configs:
        guided_labels.append(sc["label"].replace("_", "\n"))
        guided_ratios.append(results[sc["label"]]["rmsd"] / results["baseline"]["rmsd"])
    bars2 = ax2.bar(range(len(guided_labels)), guided_ratios, color="#FF9800", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(guided_labels)))
    ax2.set_xticklabels(guided_labels, fontsize=8)
    ax2.set_ylabel("RMSD ratio (guided / baseline)")
    ax2.set_title("Guidance effectiveness (lower = better)")
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="no improvement")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, guided_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # --- Panel 3: J_mean over reverse steps ---
    ax3 = fig.add_subplot(gs[1, 0])
    for sc in sweep_configs:
        label = sc["label"]
        glog = results[label].get("guidance_log", [])
        if glog:
            ts = [entry["t"] for entry in glog]
            j_means = [entry["J_mean"] for entry in glog]
            ax3.plot(ts, j_means, marker=".", markersize=3, label=label, alpha=0.8)
    ax3.set_xlabel("t (1=noise → 0=data)")
    ax3.set_ylabel("J_mean (energy)")
    ax3.set_title("Energy J over reverse trajectory")
    ax3.legend(fontsize=7)
    ax3.invert_xaxis()

    # --- Panel 4: guidance norm over reverse steps ---
    ax4 = fig.add_subplot(gs[1, 1])
    for sc in sweep_configs:
        label = sc["label"]
        glog = results[label].get("guidance_log", [])
        if glog:
            ts = [entry["t"] for entry in glog]
            g_rot = [entry["g_rot_norm"] for entry in glog]
            g_trans = [entry["g_trans_norm"] for entry in glog]
            ax4.plot(ts, g_trans, marker=".", markersize=3, label=f"{label} |g_trans|", alpha=0.8)
    ax4.set_xlabel("t (1=noise → 0=data)")
    ax4.set_ylabel("|g_trans| norm")
    ax4.set_title("Guidance translation norm over trajectory")
    ax4.legend(fontsize=7)
    ax4.invert_xaxis()

    fig.suptitle(f"GGPS SimMC Guidance on FoldFlow++ (length={LENGTH}, motif={list(motif_indices_np)}, num_t={NUM_T})",
                 fontsize=12, fontweight="bold")
    out_path = os.path.join(OUT_DIR, "guidance_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
