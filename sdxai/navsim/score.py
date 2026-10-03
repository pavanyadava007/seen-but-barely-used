"""(NAVSIM env) Official one-stage PDM scoring, with one robustness fix applied at runtime (devkit untouched):
create_scene_aggregators() fails on an empty adjacency mapping (no two time-adjacent scenes, e.g. small or
scattered subsets). In that case the two-frame extended comfort term is left NaN, which the devkit's own
compute_final_scores() already treats as "not available" (weight 0). Otherwise the original is used.
Usage: identical CLI to navsim/planning/script/run_pdm_score_one_stage.py."""
import numpy as np

import navsim.planning.script.run_pdm_score_one_stage as m

_orig = m.create_scene_aggregators


def _safe(all_mappings, full_score_df, proposal_sampling):
    if all_mappings:
        return _orig(all_mappings, full_score_df, proposal_sampling)
    df = full_score_df.copy()
    df["two_frame_extended_comfort"] = np.nan
    return df.drop(columns=["ego_simulated_states"])


m.create_scene_aggregators = _safe

if __name__ == "__main__":
    m.main()
