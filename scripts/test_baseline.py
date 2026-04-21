"""Tiny sanity check: unconditional FoldFlow++ inference with GuidedSampler (no guidance).

Goal: confirm the model loads, reverse ODE runs end-to-end, we get a backbone.
Keeps num_t small (10) and length small (30) to minimize GPU load.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import compose, initialize_config_dir

from foldflow.guidance.guided_sampler import GuidedSampler


def main():
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "runner", "config")
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="inference")

    cfg.inference.gpu_id = 1  # idle GPU
    cfg.inference.flow.num_t = 10
    cfg.inference.samples.samples_per_length = 1

    print("Building GuidedSampler...")
    sampler = GuidedSampler(cfg)
    print("Running baseline (no guidance) sampling, length=30, num_t=10 ...")
    out = sampler.sample(sample_length=30, guidance=None, verbose=False)
    print("prot_traj shape:", out["prot_traj"].shape)
    print("rigid_traj shape:", out["rigid_traj"].shape)
    print("OK: baseline sampling runs end-to-end.")


if __name__ == "__main__":
    main()
