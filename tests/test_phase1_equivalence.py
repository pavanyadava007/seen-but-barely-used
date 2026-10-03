"""Phase 1 acceptance: analysis-time code reproduces upstream outputs.

1. compiled deformable-aggregation op == upstream pure-PyTorch sampling path (module level)
2. sdxai.planner (prepare/layers/commit, no-op intervention) == upstream MotionPlanningHead.forward
3. eager attention (fp32) on the planner vs flash (fp16): final plan L2 diff < 1e-3 m
4. snapshot/restore of temporal state makes a perturbation pass invisible to the stream
"""
import numpy as np
import torch

from sdxai import planner as PL
from sdxai.eager_attention import use_eager_attention
from sdxai.runner import restore_state, snapshot_state, reset_state

N_FRAMES = 8


def _frames(runner, n=N_FRAMES):
    reset_state(runner.model)
    for i in range(n):
        data = runner.batch(i)
        metas = runner.metas_of(data)
        with torch.no_grad():
            fm = runner.features(data["img"])
            det, mp = runner.perception(fm, metas)
        yield i, data, metas, fm, det, mp


def _upstream_planner(runner, det, mp, fm, metas):
    return runner.planner_upstream(det, mp, fm, metas)


def _mine(runner, det, mp, fm, metas, iv=None, eager=None):
    h = runner.head
    P = PL.prepare(h.motion_plan_head, det, mp, fm, metas, h.det_head.anchor_encoder,
                   h.det_head.instance_bank.mask, h.det_head.instance_bank.anchor_handler)
    outs = PL.layers(h.motion_plan_head, P, iv, eager)
    return P, outs


@torch.no_grad()
def test_daf_matches_torch_sampling(runner):
    from projects.mmdet3d_plugin.ops import feature_maps_format
    for i, data, metas, fm, det, mp in _frames(runner, 2):
        dfa = [m for n, m in runner.head.det_head.named_modules()
               if type(m).__name__ == "DeformableFeatureAggregation"][-1]
        fm_list = feature_maps_format(fm, inverse=True)[0]
        inst = det["instance_feature"]
        anchor = det["prediction"][-1]
        embed = det["anchor_embed"]
        out_daf = dfa(inst, anchor, embed, fm, metas)
        # Reference: upstream torch path, but with the kernel's rule that a sample whose
        # normalised location is outside (0, 1) contributes nothing (grid_sample would still
        # interpolate zero-padded border values). Same half-pixel convention otherwise.
        bs, na = inst.shape[:2]
        kp = dfa.kps_generator(anchor, inst)
        w = dfa._get_weights(inst, embed, metas)  # bs, na, cams, levels, pts, groups
        p2d = dfa.project_points(kp, metas["projection_mat"], metas.get("image_wh"))  # bs,cams,na,pts,2
        inside = ((p2d > 0) & (p2d < 1)).all(-1).permute(0, 2, 1, 3)  # bs,na,cams,pts
        w = w * inside[:, :, :, None, :, None].to(w)
        feats = dfa.feature_sampling(fm_list, kp, metas["projection_mat"], metas.get("image_wh"))
        ref = dfa.multi_view_level_fusion(feats, w).sum(dim=2)
        ref = torch.cat([dfa.output_proj(ref), inst], dim=-1)  # residual_mode == "cat"
        rel = (out_daf - ref).abs().max() / ref.abs().max()
        assert rel < 1e-3, f"DAF vs torch rel diff {rel:.2e}"


@torch.no_grad()
def test_planner_reimplementation_matches_upstream(runner):
    h = runner.head
    for i, data, metas, fm, det, mp in _frames(runner):
        snap = snapshot_state(runner.model)
        m_up, p_up = _upstream_planner(runner, det, mp, fm, metas)
        up_final = h.motion_plan_head.post_process(det, m_up, p_up, metas)[1][0]["final_planning"]
        q_up = snapshot_state(runner.model)["queue"]
        restore_state(runner.model, snap)
        P, outs = _mine(runner, det, mp, fm, metas)
        m_me, p_me = PL.commit(h.motion_plan_head, P, outs, det, metas)
        q_me = snapshot_state(runner.model)["queue"]
        for a, b in zip(p_up["prediction"], p_me["prediction"]):
            assert torch.equal(a, b), f"frame {i}: planning_prediction differs"
        for a, b in zip(m_up["classification"], m_me["classification"]):
            assert torch.equal(a, b), f"frame {i}: motion_classification differs"
        dec = PL.decode_plan(h.motion_plan_head, det, outs, metas)
        assert torch.equal(dec["final"].cpu(), up_final), f"frame {i}: final plan differs"
        for k in ["period", "ego_period"]:
            assert torch.equal(q_up[k], q_me[k])
        for a, b in zip(q_up["ego_feature_queue"], q_me["ego_feature_queue"]):
            assert torch.equal(a, b)


@torch.no_grad()
def test_eager_planner_matches_flash(runner):
    h = runner.head
    diffs = []
    for i, data, metas, fm, det, mp in _frames(runner):
        snap = snapshot_state(runner.model)
        P, outs = _mine(runner, det, mp, fm, metas)
        flash_final = PL.decode_plan(h.motion_plan_head, det, outs, metas)["final"]
        with use_eager_attention(runner.model, prefixes=["head.motion_plan_head"]) as eag:
            outs_e = PL.layers(h.motion_plan_head, P, None, eag)
        eager_final = PL.decode_plan(h.motion_plan_head, det, outs_e, metas)["final"]
        diffs.append((flash_final - eager_final).norm(dim=-1).max().item())
        PL.commit(h.motion_plan_head, P, outs, det, metas)
    assert max(diffs) < 1e-3, f"eager vs flash max plan diff {max(diffs):.2e} m"


@torch.no_grad()
def test_noop_interventions_are_identity(runner):
    h = runner.head
    for i, data, metas, fm, det, mp in _frames(runner, 4):
        P, outs = _mine(runner, det, mp, fm, metas)
        with use_eager_attention(runner.model, prefixes=["head.motion_plan_head"]) as eag:
            base = PL.layers(h.motion_plan_head, P, None, eag)
            # masking an instance that is not among the selected keys changes nothing
            not_sel = [j for j in range(900) if j not in set(base["sel_idx"][0].tolist())][0]
            o2 = PL.layers(h.motion_plan_head, P, PL.Intervention(mask_keys=[not_sel]), eag)
        assert torch.allclose(base["planning_prediction"][-1], o2["planning_prediction"][-1], atol=1e-6)
        d0 = PL.decode_plan(h.motion_plan_head, det, outs, metas)
        d1 = PL.decode_plan(h.motion_plan_head, det, outs, metas, rescore_drop=[])
        assert torch.equal(d0["final"], d1["final"])
        PL.commit(h.motion_plan_head, P, outs, det, metas)


def _state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, list):
        return len(a) == len(b) and all(_state_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_state_equal(a[k], b[k]) for k in a)
    return a is b or a == b


@torch.no_grad()
def test_snapshot_restore_isolates_perturbation(runner):
    """A perturbed pass run under snapshot/restore leaves the temporal state bit-identical.
    (End-to-end reruns are NOT bit-identical: the deformable-aggregation kernel accumulates with
    atomicAdd, which reorders instance slots and can flip the planning mode, so isolation is
    tested on the state itself.)"""
    h = runner.head
    for i, data, metas, fm, det, mp in _frames(runner, 3):
        P, outs = _mine(runner, det, mp, fm, metas)
        PL.commit(h.motion_plan_head, P, outs, det, metas)
        before = snapshot_state(runner.model)
        fm2 = runner.features(data["img"] * 0.0)
        det2, mp2 = runner.perception(fm2, metas)
        P2, outs2 = _mine(runner, det2, mp2, fm2, metas)
        PL.commit(h.motion_plan_head, P2, outs2, det2, metas)
        restore_state(runner.model, before)
        after = snapshot_state(runner.model)
        for part in ["det_bank", "map_bank", "queue"]:
            for k in before[part]:
                assert _state_equal(before[part][k], after[part][k]), f"{part}.{k} not restored"
