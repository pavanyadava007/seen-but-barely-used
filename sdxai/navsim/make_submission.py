"""(NAVSIM env) Wrap {token: poses (8, 3)} from run_sparsedrive.py into the devkit submission format
{"first_stage_predictions": [{token: Trajectory}], "second_stage_predictions": [{}]} (one-stage scoring)."""
import pickle
import sys

import numpy as np
from navsim.common.dataclasses import Trajectory

src, dst = sys.argv[1], sys.argv[2]
d = pickle.load(open(src, "rb"))
first = {t: Trajectory(poses=np.asarray(p, np.float32)) for t, p in d["poses"].items()}
pickle.dump({"first_stage_predictions": [first], "second_stage_predictions": [{}],
             "team_name": "sdxai", "authors": ["sdxai"], "email": "", "institution": "", "country / region": ""},
            open(dst, "wb"))
print(f"{len(first)} trajectories -> {dst}")
