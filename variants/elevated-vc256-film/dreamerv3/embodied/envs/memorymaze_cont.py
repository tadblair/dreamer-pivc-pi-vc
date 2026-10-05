"""Continuous-action Memory Maze with only the egocentric image as input."""
import ctypes
import gc
import sys
from embodied.envs.from_dm import FromDM


class MemoryMazeContinuous(FromDM):

  def __init__(self, task, seed=0, size=64):
    from memory_maze import tasks
    if task not in ('9x9', '11x11', '13x13', '15x15'):
      raise ValueError(task)
    env = getattr(tasks, f'memory_maze_{task}')(
        discrete_actions=False, image_only_obs=True,
        global_observables=False, target_color_in_image=True,
        camera_resolution=size, seed=seed)
    super().__init__(env, obs_key='image')
    self._trim = ctypes.CDLL('libc.so.6').malloc_trim if sys.platform == 'linux' else None

  def step(self, action):
    obs = super().step(action)
    if obs['is_first']:
      # Recompiling a maze frees large native allocations. Return those pages
      # after reset instead of retaining them in every renderer's allocator.
      gc.collect()
      if self._trim:
        self._trim(0)
    return obs

  def close(self):
    self._env.close()
