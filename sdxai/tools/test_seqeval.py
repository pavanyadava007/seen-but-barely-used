"""Run upstream tools/test.py with the vector-map evaluation forced to its sequential path.
Upstream VectorEvaluate uses multiprocessing.Pool(fork) when a class has >81 samples (never on mini,
always on trainval); forked after CUDA/OpenMP init it deadlocks (observed: 4 h at 0% CPU on futex).
The sequential branch is the same code path upstream takes for small splits, so metrics are unchanged."""
import runpy
import sys

from projects.mmdet3d_plugin.datasets.evaluation.map import vector_eval as ve

_init = ve.VectorEvaluate.__init__


def _init_seq(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    self.n_workers = 0  # dataloader workers above are unaffected; only the eval Pool is disabled


ve.VectorEvaluate.__init__ = _init_seq
sys.argv = ["tools/test.py"] + sys.argv[1:]
runpy.run_path("tools/test.py", run_name="__main__")
