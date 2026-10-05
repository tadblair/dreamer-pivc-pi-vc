from pathlib import Path
import tempfile,json,pickle,hashlib
import numpy as np
from balanced_session_criterion import BalancedSessionCriterion
from elevated_track_env import ElevatedTrack
def tran(**overrides):
 d={'is_first':False,'is_last':True,'log/trials_completed':32,'log/session_number':1,'log/session_steps':7000,'log/fell':0,'log/prox_choices':16,'log/dist_choices':16,'log/prox_correct':12,'log/dist_correct':12};d.update(overrides);return d
with tempfile.TemporaryDirectory() as tmp:
 c=BalancedSessionCriterion(tmp)
 c.observe(tran(**{'is_last':False,'log/trials_completed':0}),0,5);assert c.first_reward is None
 c.observe(tran(**{'is_last':False,'log/trials_completed':1,'log/session_steps':80}),3,1283);assert c.first_reward['global_step']==1283
 for override in [{'log/prox_correct':11,'log/dist_correct':16},{'log/prox_correct':16,'log/dist_correct':11},{'log/prox_choices':15},{'log/trials_completed':31},{'log/fell':1}]:
  c.observe(tran(**override),0,9000);assert not c.solved
 saved=pickle.loads(pickle.dumps(c.save()));c=BalancedSessionCriterion(tmp);c.load(saved)
 c.observe(tran(),2,10000);assert c.solved and c.success['steps_since_first_reward']==8717
 c.observe(tran(),1,10001);assert c.success['global_step']==10000
 (Path(tmp)/'ckpt').mkdir();(Path(tmp)/'ckpt/latest').write_text('verified-test')
 c.finish(10000);assert json.loads((Path(tmp)/'criterion_reached.json').read_text())['first_reward']['global_step']==1283
# Check actual per-start classification follows the trial's sequence entry.
e=ElevatedTrack(seed=9,always_dim_east=True,delayed_reward_input=True);o=e.reset();s=e.session
s.sequence=np.array([1,-1]*16);s.trial_choices=[dict(trial=i+1,corner=0 if i//2<12 else 1,step=i+1) for i in range(32)]
s.completed=32;s.choices=32;s.correct=24
p0,heading=e.pose();o=e.observe(0,True,np.zeros(2,np.float32),None)
assert o['log/prox_correct']==12 and o['log/dist_correct']==12 and o['log/prox_choices']==16 and o['log/dist_choices']==16
for available,brightness in [(True,50),(False,50),(True,50)]:
 s.available=available;e.sync();np.testing.assert_allclose(e.physics.model.mat_rgba[e.screen_id,:3],brightness/(255*1000))
e.reset();s=e.session;s.config=2;e.sync();e.physics.data.qpos[:]=[1.18,1.18,0,0];e.physics.data.qvel[:]=0;e.physics.forward()
o=e.step(dict(reset=False,action=np.zeros(2,np.float32)));assert o['reward']==1 and o['reward_received_previous'][0]==0
assert e.step(dict(reset=False,action=np.zeros(2,np.float32)))['reward_received_previous'][0]==1
assert e.reset()['reward_received_previous'][0]==0;e.close()
print(json.dumps(dict(passed=True,balanced_threshold_boundary=True,imbalance_rejected=True,first_reward_persistence=True,dim_cue_and_delayed_reward=True,historical_checkpoint_checks_omitted=True)))
