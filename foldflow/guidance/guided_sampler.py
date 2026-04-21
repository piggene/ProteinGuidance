"""Minimal guided sampler for FoldFlow++.

Builds the FF2 model + train.Experiment (same as runner.inference.Sampler) but
skips ESMFold/ProteinMPNN so test-time guidance can be driven without heavy
evaluators. For sanity checks we only need the flow sampler and the final
atom37 trajectory.
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
import tree
from omegaconf import DictConfig, OmegaConf

from foldflow.data import utils as du
from foldflow.guidance.guided_inference import guided_inference_fn
from foldflow.models.ff2flow.ff2_dependencies import FF2Dependencies
from foldflow.models.ff2flow.flow_model import FF2Model
from runner import train


class GuidedSampler:
    def __init__(self, conf: DictConfig, device: Optional[str] = None):
        self._log = logging.getLogger(__name__)
        OmegaConf.set_struct(conf, False)
        self._conf = conf
        self._infer_conf = conf.inference
        self._fm_conf = self._infer_conf.flow
        self._sample_conf = self._infer_conf.samples

        if device is None:
            if torch.cuda.is_available():
                gpu_id = self._infer_conf.gpu_id if self._infer_conf.gpu_id is not None else 0
                self.device = f"cuda:{gpu_id}"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._log.info(f"GuidedSampler using device: {self.device}")

        weights_pkl = du.read_pkl(
            self._infer_conf.weights_path, use_torch=True, map_location=self.device
        )
        if conf.model.model_name == "ff1":
            raise NotImplementedError("GuidedSampler currently targets FF2.")
        deps = FF2Dependencies(conf)
        self.model = FF2Model.from_ckpt(weights_pkl, deps).to(self.device)
        self.model.eval()
        self.flow_matcher = deps.flow_matcher
        self.exp = train.Experiment(conf=self._conf, model=self.model)

    @torch.no_grad()
    def sample(
        self,
        sample_length: int,
        fixed_mask: Optional[np.ndarray] = None,
        motif_rigids_impute: Optional[torch.Tensor] = None,
        guidance=None,
        num_t: Optional[int] = None,
        noise_scale: Optional[float] = None,
        t_start: Optional[float] = None,
        init_rigids: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict:
        """Sample one protein of length `sample_length`.

        Args:
            sample_length: length N.
            fixed_mask: [N] np.ndarray, 1 where residue is fixed motif, 0 scaffold.
            motif_rigids_impute: [N, 7] torch tensor of motif target frames placed
              at motif positions (zero elsewhere). Motif positions remain at these
              values throughout sampling because reverse() is masked with flow_mask.
            guidance: optional SimMCGuidance / MCGuidance instance.
            t_start: if set, start reverse from this time instead of 1.0.
                Used for iterative refinement.
            init_rigids: [N, 7] pre-computed initial rigids (e.g. re-noised from
                previous round). Overrides sample_ref when provided.
        Returns: dict from guided_inference_fn (prot_traj, rigid_traj, ...).
        """
        res_mask = np.ones(sample_length)
        if fixed_mask is None:
            fixed_mask = np.zeros_like(res_mask)
        aatype = torch.zeros(sample_length, dtype=torch.int32)
        chain_idx = torch.zeros_like(aatype)

        if init_rigids is not None:
            # Use pre-computed rigids (e.g. re-noised from previous round)
            ref_sample = {"rigids_t": init_rigids.detach().cpu()}
        else:
            ref_sample = self.flow_matcher.sample_ref(
                n_samples=sample_length, as_tensor_7=True
            )  # dict with rigids_t: torch tensor [N, 7]

        # Plant motif frames at motif positions so they survive masking.
        if motif_rigids_impute is not None and fixed_mask.sum() > 0:
            impute_cpu = motif_rigids_impute.detach().cpu()
            rigids_t = ref_sample["rigids_t"].clone().to(impute_cpu.dtype)
            fm_t = torch.tensor(fixed_mask, dtype=impute_cpu.dtype).unsqueeze(-1)
            rigids_t = rigids_t * (1 - fm_t) + impute_cpu * fm_t
            ref_sample["rigids_t"] = rigids_t

        res_idx = torch.arange(1, sample_length + 1)
        init_feats = {
            "res_mask": res_mask,
            "seq_idx": res_idx,
            "fixed_mask": fixed_mask,
            "torsion_angles_sin_cos": np.zeros((sample_length, 7, 2)),
            "sc_ca_t": np.zeros((sample_length, 3)),
            "aatype": aatype,
            "chain_idx": chain_idx,
            **ref_sample,
        }
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(lambda x: x[None].to(self.device), init_feats)

        sample_out = guided_inference_fn(
            self.exp,
            init_feats,
            guidance=guidance,
            num_t=num_t if num_t is not None else self._fm_conf.num_t,
            min_t=self._fm_conf.min_t,
            t_start=t_start,
            aux_traj=True,
            noise_scale=noise_scale
            if noise_scale is not None
            else self._fm_conf.noise_scale,
            verbose=verbose,
        )
        return tree.map_structure(
            lambda x: x[:, 0] if hasattr(x, "ndim") and x.ndim > 1 else x, sample_out
        )
