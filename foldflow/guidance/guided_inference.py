"""Guided inference loop for FoldFlow++.

Mirror of `runner.train.Experiment.inference_fn` with a guidance hook inserted
between the model forward pass and the flow_matcher reverse step. Pure
test-time: no training code is modified.
"""

import copy

import numpy as np
import torch

from foldflow.data import utils as du
from foldflow.data import all_atom
from openfold.utils import rigid_utils as ru


def guided_inference_fn(
    exp,
    data_init,
    guidance=None,
    num_t=None,
    min_t=None,
    t_start=None,
    center=True,
    aux_traj=False,
    self_condition=True,
    noise_scale=1.0,
    context=None,
    verbose=False,
):
    """Run reverse flow with optional test-time guidance.

    Args:
        exp: `runner.train.Experiment` instance (pre-loaded model + flow_matcher).
        data_init: dict of torch tensors from Sampler.sample (batched, on device).
        guidance: object exposing
            .compute(rigids_t, rigid_pred, t, flow_mask) -> (g_rot, g_trans, info)
          with g_rot: [B,N,3,3] and g_trans: [B,N,3]. Pass None for unconditional.
        t_start: if set, start reverse from this time instead of 1.0.
            Used for iterative refinement (re-noising to t_start < 1).

    Returns: same dict as exp.inference_fn.
    """
    sample_feats = copy.deepcopy(data_init)
    device = sample_feats["rigids_t"].device
    if sample_feats["rigids_t"].ndim == 2:
        t_placeholder = torch.ones((1,)).to(device)
    else:
        t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(device)
    if num_t is None:
        num_t = exp._data_conf.num_t
    if min_t is None:
        min_t = exp._data_conf.min_t
    if t_start is None:
        t_start = 1.0
    reverse_steps = np.linspace(min_t, t_start, num_t)[::-1]
    dt = reverse_steps[0] - reverse_steps[1]

    all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
    all_bb_prots = []
    all_trans_0_pred = []
    all_bb_0_pred = []
    guidance_log = []

    with torch.no_grad():
        if exp._model_conf.embed.embed_self_conditioning and self_condition:
            sample_feats = exp._set_t_feats(
                sample_feats, reverse_steps[0], t_placeholder
            )
            sample_feats = exp._self_conditioning(sample_feats)

        for t in reverse_steps:
            sample_feats = exp._set_t_feats(sample_feats, t, t_placeholder)
            model_out = exp.model(sample_feats)
            rot_vectorfield = model_out["rot_vectorfield"]  # [B, N, 3, 3]
            trans_vectorfield = model_out["trans_vectorfield"]  # [B, N, 3]
            rigid_pred = model_out["rigids"]  # [B, N, 7]
            if exp._model_conf.embed.embed_self_conditioning:
                sample_feats["sc_ca_t"] = rigid_pred[..., 4:]

            fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
            flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

            # === Guidance hook ===
            if guidance is not None:
                g_rot, g_trans, info = guidance.compute(
                    rigids_t=sample_feats["rigids_t"],
                    rigid_pred=rigid_pred,
                    t=float(t),
                    flow_mask=flow_mask,
                )
                rot_vectorfield = rot_vectorfield + g_rot.to(rot_vectorfield.dtype)
                trans_vectorfield = trans_vectorfield + g_trans.to(
                    trans_vectorfield.dtype
                )
                if verbose:
                    print(f"[guidance t={float(t):.3f}] {info}")
                guidance_log.append({"t": float(t), **info})

            rots_t, trans_t, rigids_t = exp.flow_matcher.reverse(
                rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                rot_vectorfield=du.move_to_np(rot_vectorfield),
                trans_vectorfield=du.move_to_np(trans_vectorfield),
                flow_mask=du.move_to_np(flow_mask),
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale,
            )

            sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
            if aux_traj:
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

            gt_trans_0 = sample_feats["rigids_t"][..., 4:]
            pred_trans_0 = rigid_pred[..., 4:]
            trans_pred_0 = (
                flow_mask[..., None] * pred_trans_0
                + fixed_mask[..., None] * gt_trans_0
            )
            psi_pred = model_out["psi"]
            if aux_traj:
                atom37_0 = all_atom.compute_backbone(
                    ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                )[0]
                all_bb_0_pred.append(du.move_to_np(atom37_0))
                all_trans_0_pred.append(du.move_to_np(trans_pred_0))
            atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
            all_bb_prots.append(du.move_to_np(atom37_t))

    flip = lambda x: np.flip(np.stack(x), (0,))
    all_bb_prots = flip(all_bb_prots)
    if aux_traj:
        all_rigids = flip(all_rigids)
        all_trans_0_pred = flip(all_trans_0_pred)
        all_bb_0_pred = flip(all_bb_0_pred)

    ret = {"prot_traj": all_bb_prots}
    if aux_traj:
        ret["rigid_traj"] = all_rigids
        ret["trans_traj"] = all_trans_0_pred
        ret["psi_pred"] = psi_pred[None]
        ret["rigid_0_traj"] = all_bb_0_pred
    if guidance is not None:
        ret["guidance_log"] = guidance_log
    return ret
