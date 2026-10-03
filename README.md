# Seen but Barely Used

Code for **"Seen but Barely Used: Concept-Level Interpretability of Pedestrian Perception in a Sparse End-to-End
Driving Model"** (Pavan Yadav Annappa). Paper: [`paper/paper.pdf`](paper/paper.pdf).

The study opens the object queries of [SparseDrive](https://github.com/swc-17/SparseDrive) and traces pedestrians
from image evidence to the planner: attribution, linear probes with control tasks, TCAV, sparse autoencoders,
confound-controlled concept tests, and causal interventions that close every planner-side route (attention,
collision rescoring, temporal memory). Experiments use only public code and data: nuScenes v1.0-trainval,
NAVSIM / OpenScene and Bench2Drive / CARLA.

## Layout

| Path | Content |
|---|---|
| `sdxai/` | analysis package (extraction, attribution, concepts/SAE, interventions, NAVSIM and CARLA runners, report) |
| `scripts_xai/` | phase scripts `p0`-`p7` (baseline, extraction, instances, attribution, concepts, interventions, temporal, report) |
| `tests/` | unit tests |
| `paper/` | LaTeX source, generated tables, figures, and the scripts that compute every number from the result files |

## Setup

1. Clone upstream SparseDrive (tested at commit `fbadea693cbef3f3daea7705e521c0dd3321605c`), install it as described
   there (PyTorch 1.13.1, mmcv-full 1.7.1, flash-attention), and download the released stage-2 checkpoint.
2. Copy `sdxai/`, `scripts_xai/` and `tests/` of this repository into the SparseDrive root. No upstream file is modified.
3. Set the paths:
   ```bash
   export SDXAI_ROOT=/path/to/workspace        # contains SparseDrive/, nuscenes/, navsim_exp/, b2d/
   export OPENSCENE_DATA_ROOT=/path/to/openscene   # NAVSIM / OpenScene data
   export SDXAI_VERSION=trainval               # or mini for a quick check
   ```
4. Run the phases in order (`scripts_xai/p0_*.sh` ... `p7_report.sh`); all long runs are resumable.
5. `paper/build.sh` regenerates the numbers, tables, figures and PDF from the results.

## Licence

MIT (see `LICENSE`). Builds on SparseDrive (MIT, Copyright (c) 2024 swc-17). Figures contain nuScenes camera images
(Motional), used under CC BY-NC-SA 4.0.
