import tempfile,pickle,json
from pathlib import Path
import elements
from verified_checkpoint import VerifiedCheckpoint
with tempfile.TemporaryDirectory() as d:
    root=Path(d)
    c=VerifiedCheckpoint(str(root));c.step=elements.Counter(3000000);c.agent=elements.Counter(7)
    c.save(str(root/'pivc_custom_name'))
    (root/'latest').write_text('pivc_custom_name')
    c.step.increment(100)
    c.save()
    latest=(root/'latest').read_text().strip()
    assert latest!='pivc_custom_name' and (root/latest/'agent.pkl').exists()
    assert pickle.loads((root/latest/'step.pkl').read_bytes())==3000100
    assert not (root/'pivc_custom_name').exists()
    c.step.increment(100);c.save()
    latest=(root/'latest').read_text().strip()
    assert pickle.loads((root/latest/'step.pkl').read_bytes())==3000200
    assert len(list(root.iterdir()))==2
    restored=VerifiedCheckpoint(str(root));restored.step=elements.Counter();restored.agent=elements.Counter();restored.load_or_save()
    assert int(restored.step)==3000200 and int(restored.agent)==7
print(json.dumps(dict(passed=True,custom_name_cleanup=True,second_save=True,retains_one=True,reload=True)))
