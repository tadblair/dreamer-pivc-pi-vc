"""Previous-image visual features conditioned on measured transition signals."""
import jax.numpy as jnp
from embodied.jax import nets as nn
from .rssm import Encoder


class SensoryEncoder(Encoder):
    # Ninjax discovers configuration fields on the concrete class.
    units: int = 512
    norm: str = 'rms'
    act: str = 'silu'
    depth: int = 32
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kw):
        # No current-image or vector bypass into the visual encoder.
        super().__init__({'image': obs_space['source_image']}, **kw)

    def __call__(self, carry, obs, reset, training, single=False):
        carry, entries, visual = super().__call__(
            carry, {'image': obs['source_image']}, reset, training, single)
        embeddings = []
        for key, width in [('vibrissal', 128), ('efference', 32), ('motion', 32)]:
            x = nn.cast(obs[key], force=True)
            x = self.sub(f'film_{key}', nn.Linear, width, **self.kw)(x)
            embeddings.append(nn.act('silu')(x))
        x = jnp.concatenate(embeddings, -1)
        for i in range(2):
            x = nn.act('silu')(self.sub(f'film_hidden{i}', nn.Linear, 256, **self.kw)(x))
        # Zero final weights and biases yield exact identity at initialization.
        kw = dict(self.kw, outscale=0.0)
        scale = self.sub('film_scale', nn.Linear, visual.shape[-1], **kw)(x)
        shift = self.sub('film_shift', nn.Linear, visual.shape[-1], **kw)(x)
        sensory = (1 + scale) * visual + shift
        # One observed binary feature, appended AFTER FiLM, no embedding.
        received = nn.cast(obs['reward_received_previous'], force=True)
        received = jnp.where(reset[..., None], 0, received)
        return carry, entries, jnp.concatenate([sensory, received], -1)
