"""(NAVSIM env) Replay agent: returns trajectories precomputed by run_sparsedrive.py, keyed by scene token.

Lets the official one-stage PDM scoring (run_pdm_score_one_stage.py) score SparseDrive, which runs in a
separate python 3.8 / torch 1.13 environment. No sensors are loaded on the NAVSIM side."""
import pickle

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class PrecomputedAgent(AbstractAgent):
    requires_scene = True

    def __init__(self, trajectory_file: str, name: str = "SparseDrive",
                 trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)):
        super().__init__(trajectory_sampling, requires_scene=True)
        self._file = trajectory_file
        self._name = name
        self._poses = None

    def name(self) -> str:
        return self._name

    def initialize(self) -> None:
        self._poses = pickle.load(open(self._file, "rb"))["poses"]

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene = None) -> Trajectory:
        if self._poses is None:
            self.initialize()
        token = scene.scene_metadata.initial_token
        if token not in self._poses:
            raise KeyError(f"no precomputed trajectory for token {token}")
        return Trajectory(np.asarray(self._poses[token], np.float32), self._trajectory_sampling)
