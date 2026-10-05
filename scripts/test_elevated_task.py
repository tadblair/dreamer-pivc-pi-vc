"""Deterministic rule boundaries and real-physics integration checks."""
import json
from pathlib import Path
import numpy as np
from elevated_session import Session,start_sequence
from elevated_track_env import ElevatedTrack
from elevated_track_barriers import CONFIGS

rng=np.random.RandomState(41)
for _ in range(1000):
 seq=start_sequence(rng)
 assert (seq==1).sum()==16 and (seq==-1).sum()==16
 assert not any(np.all(seq[i:i+4]==seq[i]) for i in range(29))
seq=np.array([1,-1]*16)
for side,expected in [(1,2),(-1,1)]:
 s=Session(rng,seq*side)
 for step in range(39):s.begin_step();s.sample([0,.9]);s.end_step();assert s.config==0
 s.begin_step();s.sample([0,.9],fraction=.5);assert s.config==0
 s.sample([0,.8]);assert s.config==0
 s.sample([0,.80001]);assert s.config==expected
 assert s.sample([-.5,.5])==0
# Wrong corner counts once but leaves reward available; subsequent correct corner pays.
s=Session(rng,seq);s.begin_step()
assert s.sample([-1.18,1.18])==0 and s.choices==1 and s.correct==0
assert s.sample([-1.18,-1.18])==0 and s.choices==1
assert s.sample([1.18,1.18])==1 and s.config==4 and not s.available
assert s.sample([1.18,1.18])==0
s.end_step();s.begin_step();s.previous_y=-.19
s.sample([-.5,-.21]);assert s.config==6
s.end_step();s.begin_step();s.sample([0,-.7]);assert s.config==0 and s.available and s.choice is None
# Positive-x config4 branch and config3 return, plus terminal 32nd reward.
s=Session(rng,-seq);s.begin_step();assert s.sample([1.18,1.18])==1 and s.config==3
s.end_step();s.begin_step();s.sample([0,.79]);assert s.config==0
s.config=4;s.transitioned=False;s.previous_y=-.2;s.sample([.3,-.20001]);assert s.config==5
s.end_step();s.begin_step();s.sample([0,-.79]);assert s.config==0
s.completed=31;s.begin_step();assert s.sample([1.18,1.18])==1
assert s.completed==32 and s.done and s.terminal and not s.available
s=Session(rng,seq);s.steps=7679;s.begin_step();s.sample([0,.25]);s.end_step()
assert s.done and not s.terminal and s.steps==7680
s=Session(rng,seq);s.begin_step();assert s.sample([.5,.5],False)==-20 and s.done and s.fell
assert CONFIGS[6]==[1,5,7,9,11,13,15,16]

# Physical checks: collision blocks locomotion rather than correcting coordinates afterward.
e=ElevatedTrack(seed=1)
a=lambda v:dict(action=np.array(v,np.float32),reset=False)
obs=e.reset();assert obs['image'].shape==(64,64,2)
for v in obs.values():assert np.isfinite(v).all()
# Force prox start, face south, drive into raised B9 for 5 seconds.
e.session=Session(rng,seq)
e.physics.data.qpos[:]=[0,.25,0,-np.pi/2];e.physics.data.qvel[:]=0;e.physics.forward();e.sync()
for _ in range(20):
 obs=e.step(a([1,0]));assert not obs['is_last']
assert e.pose()[0][1]>.210,(e.pose(),e.physics.data.qvel)
blocked_y=float(e.pose()[0][1])
# Lower B9 (config2) and confirm same force controller traverses its old plane.
e.session.config=2;e.sync()
for _ in range(8):obs=e.step(a([1,0]))
assert e.pose()[0][1]<.15,e.pose()
# Full-throttle long collision cannot pass closed B7.
for _ in range(20):obs=e.step(a([1,0]));assert not obs['is_last']
assert e.pose()[0][1]>-.18,e.pose()
# Real motion into a reward circle, screen dims, no repeat reward.
e.reset();e.session=Session(rng,seq);e.session.config=2;e.sync()
e.physics.data.qpos[:]=[1.13,1.13,0,np.pi/4];e.physics.data.qvel[:]=0;e.physics.forward()
rewards=[]
for _ in range(3):rewards.append(float(e.step(a([1,0]))['reward']))
assert sum(rewards)==1,rewards
assert e.session.config==4 and not e.session.available
np.testing.assert_allclose(e.physics.model.mat_rgba[e.screen_id,:3],50/(255*1000))
e.session.config=6;e.session.transitioned=False;e.physics.data.qpos[:]=[0,-.7,0,np.pi/2];e.physics.data.qvel[:]=0;e.physics.forward()
e.step(a([0,0]));assert e.session.config==0 and e.session.available
np.testing.assert_allclose(e.physics.model.mat_rgba[e.screen_id,:3],255/(255*1000))
# Start just beyond support; terminate at first substep with exactly -20.
e.physics.data.qpos[:]=[.5,.5,0,0];e.physics.data.qvel[:]=0;e.physics.forward()
obs=e.step(a([0,0]));assert obs['reward']==-20 and obs['is_terminal'] and obs['is_last']
# Many independent sessions reset all state, preserve finite sensory arrays and stable rest.
for _ in range(4):
 obs=e.reset()
 for _ in range(10):obs=e.step(a([0,0]))
 assert not obs['is_last'] and abs(e.pose()[0][2]-1.0185)<.003
 assert np.isfinite(obs['motion']).all() and np.isfinite(obs['vibrissal']).all()
# Save reviewable physical scene and stereo reset views.
from PIL import Image
out=Path('artifacts/elevated-task');out.mkdir(parents=True,exist_ok=True)
(out/'active-scene.xml').write_text(e.scene_xml)
image=e.reset()['image'];Image.fromarray(np.concatenate([image[:,:,0],image[:,:,1]],axis=1)).resize((1024,512)).save(out/'stereo_start.png')
report=dict(passed=True,balanced_sequences_checked=1000,config0_wait_seconds=10,
            collision_blocked_y=blocked_y,real_reward_collection=True,screen_toggle=True,
            exact_fall_penalty=-20,time_limit=7680,max_rewards=32)
(out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
e.close()
