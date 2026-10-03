"""Write a NAVSIM train_test_split (+ scene_filter) config listing exactly the scenes in a trajectory file
(run_sparsedrive.py output), so one-stage PDM scoring evaluates only scenes SparseDrive produced.
Filter settings (4 history / 10 future frames, has_route) are those of navtest/navmini."""
import os
import pickle
import sys

src, name, data_split = sys.argv[1], sys.argv[2], sys.argv[3]
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "train_test_split")
d = pickle.load(open(src, "rb"))
tokens = sorted(d["poses"])
logs = sorted({m["log"] for m in d["meta"].values()})
with open(os.path.join(root, "scene_filter", f"{name}.yaml"), "w") as f:
    f.write("_target_: navsim.common.dataclasses.SceneFilter\n_convert_: 'all'\n\nnum_history_frames: 4\n"
            "num_future_frames: 10\nframe_interval: 1\nhas_route: true\n\nmax_scenes: null\nlog_names:\n")
    f.writelines(f"  - '{l}'\n" for l in logs)
    f.write("tokens:\n")
    f.writelines(f"  - '{t}'\n" for t in tokens)
with open(os.path.join(root, f"{name}.yaml"), "w") as f:
    f.write(f"defaults:\n  - scene_filter: {name}\n\ndata_split: {data_split}\n")
print(f"{name}: {len(tokens)} tokens in {len(logs)} logs")
