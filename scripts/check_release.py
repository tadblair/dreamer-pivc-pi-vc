"""Check released presets against recorded configs and inspect posterior inputs."""
import ast
import json
from pathlib import Path
import elements
from ruamel.yaml import YAML
root = Path(__file__).resolve().parents[1]
yaml = YAML(typ='safe')
for model in ('pivc256', 'pi256', 'vc256'):
    folder = root / 'variants' / f'elevated-{model}-film' / 'dreamerv3' / 'dreamerv3'
    presets = yaml.load((folder / 'configs.yaml').read_text())
    cfg = elements.Config(presets['defaults'])
    for key in ('elevated_track', 'size50m'):
        cfg = cfg.update(presets[key])
    cfg = cfg.update({'run.steps': float('inf'), 'batch_size': 4, 'batch_length': 256})
    actual = cfg.flat
    reference = elements.Config(yaml.load((root / 'reference' / model / 'config.yaml').read_text())).flat
    differences = {k: (actual.get(k), v) for k, v in reference.items() if k != 'logdir' and actual.get(k) != v}
    assert not differences, (model, differences)
    text = (folder / 'sensory_film.py').read_text()
    tree = ast.parse(text)
    keys = {n.slice.value for n in ast.walk(tree) if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == 'obs'
            and isinstance(n.slice, ast.Constant)}
    assert keys == {'source_image', 'reward_received_previous'}, keys
    assert "[('vibrissal', 128), ('efference', 32), ('motion', 32)]" in text
    assert 'option' not in text.lower()
    assert 'jnp.concatenate([deter, tokens], -1)' in (folder / 'rssm.py').read_text()
    assert actual['agent.imag_length'] == 15
    print(model + ': exact reference config match (except logdir); no option posterior input')
manifest = json.loads((root / 'reference/manifest.json').read_text())
for model, item in manifest['models'].items():
    row = item['criterion']
    assert row['criterion_met'] and row['rewards'] == 32 and not row['fell']
print('All three stored criterion records verified.')
