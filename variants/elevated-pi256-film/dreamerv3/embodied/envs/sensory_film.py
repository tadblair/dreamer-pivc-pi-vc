"""Aligned transition observations for the isolated sensory-FiLM agent."""
import functools
import elements
import numpy as np
from embodied.envs.memorymaze_cont import MemoryMazeContinuous
from embodied.envs.film_rays import RaySensors


def motion_signal(start, end, forward, next_forward):
    """Net displacement in the initial head frame; yaw is signed left-positive."""
    forward = np.asarray(forward, np.float64)
    forward = forward / np.linalg.norm(forward)
    next_forward = np.asarray(next_forward, np.float64)
    next_forward = next_forward / np.linalg.norm(next_forward)
    delta = np.asarray(end)[:2] - np.asarray(start)[:2]
    left = np.array([-forward[1], forward[0]])
    yaw = np.arctan2(forward[0] * next_forward[1] - forward[1] * next_forward[0],
                     forward @ next_forward)
    return np.asarray([delta @ forward / .2, delta @ left / .2, yaw / (np.pi / 6)], np.float32)


class SensoryMemoryMaze(MemoryMazeContinuous):
    @functools.cached_property
    def obs_space(self):
        return dict(super().obs_space,
                    source_image=elements.Space(np.uint8, (64, 64, 3)),
                    vibrissal=elements.Space(np.float32, (61,), 0, 1),
                    efference=elements.Space(np.float32, (2,), -1, 1),
                    motion=elements.Space(np.float32, (3,)))

    def _sample(self):
        physics = self._env.physics
        # Same forward axis used by Memory Maze's agent_dir observable.
        forward = np.asarray(physics.bind(self._env.task._walker.root_body).xmat).reshape(3, 3)[:2, 1].copy()
        readings = self._sensors.read(forward)
        return readings['origin'], forward, readings['activity'].astype(np.float32)

    def step(self, action):
        obs = super().step(action)
        if obs['is_first']:
            # Maze reset recompiles MuJoCo; never retain old model/data pointers.
            self._sensors = RaySensors(self._env.physics)
        position, forward, tactile = self._sample()
        if obs['is_first']:
            source = obs['image'].copy()
            source_tactile = tactile.copy()
            efference = np.zeros(2, np.float32)
            motion = np.zeros(3, np.float32)
        else:
            source, source_tactile = self._image, self._tactile
            efference = np.asarray(action['action'], np.float32).copy()
            motion = motion_signal(self._position, position, self._forward, forward)
        self._image = obs['image'].copy()
        self._tactile = tactile.copy()
        self._position, self._forward = position, forward
        return dict(obs, source_image=source, vibrissal=source_tactile,
                    efference=efference, motion=motion)
