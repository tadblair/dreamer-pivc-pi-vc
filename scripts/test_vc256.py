from pathlib import Path
import json,os,sys,pickle,hashlib
import numpy as np
from vc256_session import Session,ROTATIONS,raised_doors,return_frame
from vc256_env import ElevatedTrack
root=Path(__file__).resolve().parents[1]
existing={}  # Checkpoint identities are in reference/manifest.json.
class NextOrientation:
 def __init__(self,q):self.q=q
 def randint(self,n):return self.q
# Unit checks: all combinations of current/new orientation and next trial assignment.
count=0
for old in range(4):
 for new in range(4):
  for sign in [-1,1]:
   s=Session(np.random.RandomState(1),[sign,-sign]*16)
   s.sequence=np.array([-sign,sign]*16);s.orientation=old;s.goal_frame=ROTATIONS[old].copy();s.rng=NextOrientation(new)
   s.begin_step();reward=s.sample(s.goal_frame@np.array([1.18,1.18]))
   assert reward==1 and s.completed==1 and not s.available and s.orientation==new
   arm=ROTATIONS[new]@np.array([0,sign]);config,T=return_frame(ROTATIONS[old]@np.ones(2,dtype=int),arm)
   assert config==s.config;np.testing.assert_array_equal(T,s.return_frame)
   assert len(raised_doors(config,T))==len(raised_doors(config,np.eye(2)))
   count+=1
# Config0 timer and trigger must follow the new goal frame, both trial labels.
for q in range(4):
 for sign in [-1,1]:
  s=Session(np.random.RandomState(1),[sign,-sign]*16);s.orientation=q;s.goal_frame=ROTATIONS[q]
  for step in range(39):s.begin_step();s.sample(s.goal_frame@[0,sign*.85]);assert s.config==0
  s.begin_step();s.sample(s.goal_frame@[0,sign*.5]);assert s.config==0
  s.begin_step();s.sample(s.goal_frame@[0,sign*.85]);assert s.config==(2 if sign==1 else 1)
print('Passed 32 frame transitions and 8 rotated timer checks',flush=True)
e=ElevatedTrack(seed=9,always_dim_east=False,delayed_reward_input=True)
o=e.reset();assert o['reward_received_previous'][0]==0
original_observe=e.observe
# Physics navigation tests skip image rendering only; actions, contacts, timing, and transitions remain real.
e.observe=lambda reward,first,command,before:dict(reward=reward)
records=json.loads((root/'tests/fixtures/vc256_routes.json').read_text())
arms={'N':np.array([0,1]),'E':np.array([1,0]),'S':np.array([0,-1]),'W':np.array([-1,0])}
corners={'NE':0,'NW':1,'SW':2,'SE':3}
results=[]
def bright_indices():
 return [i for i,name in enumerate(['east','north','west','south']) if e.physics.model.mat_rgba[e.physics.model.name2id('screen_'+name,'material'),0]>.0005]
def go(target,stop_at_config0=True):
 for k in range(160):
  pos,yaw=e.pose();delta=target-pos[:2];dist=np.linalg.norm(delta)
  if dist<.015:return
  desired=np.arctan2(delta[1],delta[0]);err=np.arctan2(np.sin(desired-yaw),np.cos(desired-yaw))
  turn=np.clip(err/(np.pi*.25),-1,1)
  forward=min(.8,dist/.075) if abs(err)<.13 else 0
  e.step(dict(reset=False,action=np.array([forward,turn])))
  assert not e.session.fell,('fell',target,pos)
  if stop_at_config0 and e.session.config==0:return
 raise AssertionError(('stuck',target,e.pose(),e.session.config))
for rec in records:
 for route in rec['routes']:
  e.reset();s=e.session;old=corners[rec['previous_goal']];s.orientation=old;s.goal_frame=ROTATIONS[old].copy();s.sequence=np.array([-1,1]*16)
  new=next(i for i,Q in enumerate(ROTATIONS) if np.array_equal(Q@[0,1],arms[rec['next_start_arm']]))
  s.rng=NextOrientation(new);goal=s.goal_frame@np.array([1.18,1.18]);e.physics.data.qpos[:]=[goal[0],goal[1],0,0];e.physics.data.qvel[:]=0;e._door_key=None;e.sync();e.physics.forward()
  assert bright_indices()==[old]
  assert e.step(dict(reset=False,action=np.zeros(2)))['reward']==1
  assert bright_indices()==[] and s.config==rec['initial_config']
  saw=set([s.config])
  for p in route['points'][1:]:
   go(np.array(p));saw.add(s.config)
   if s.config==0:break
  assert s.config==0 and bright_indices()==[new],(rec,route['kind'],s.config)
  expected=3 if route['kind']=='near' else 5 if route['kind']=='short' else 6
  assert expected in saw,(route['kind'],saw)
  assert s.completed==1 and s.available
  results.append(dict(goal=rec['previous_goal'],start_arm=rec['next_start_arm'],route=route['kind'],steps=s.steps,configs=sorted(saw)))
  print('Physics route passed',results[-1],flush=True)
# Raised door physically stops forward motion in all four orientations.
for q in range(4):
 e.reset();s=e.session;s.orientation=q;s.goal_frame=ROTATIONS[q];s.config0_steps=-1000;e._door_key=None;e.sync()
 xy=s.goal_frame@[0,.10];e.physics.data.qpos[:]=[xy[0],xy[1],0,np.pi/2+q*np.pi/2];e.physics.data.qvel[:]=0;e.physics.forward()
 for _ in range(20):e.step(dict(reset=False,action=np.array([1.,0.])))
 local=e.pose()[0][:2]@s.goal_frame;assert local[1]<.195 and not s.fell,local
# End-to-end goal seeking in all rotations, from both prox and dist starts.
for q in range(4):
 for sign in [-1,1]:
  e.reset();s=e.session;s.sequence=np.array([sign,-sign]*16);s.orientation=q;s.goal_frame=ROTATIONS[q];s.return_frame=s.goal_frame.copy();s.previous_y=sign*.25
  xy=s.goal_frame@[0,sign*.25];e.physics.data.qpos[:]=[xy[0],xy[1],0,(-np.pi/2 if sign==1 else np.pi/2)+q*np.pi/2];e.physics.data.qvel[:]=0;e._door_key=None;e.sync();e.physics.forward()
  for _ in range(40):e.step(dict(reset=False,action=np.zeros(2)))
  for target in [[0,sign*.86],[0,0],[1,0],[1,1],[1.18,1.18]]:
   go(s.goal_frame@target,stop_at_config0=False)
   if s.completed:break
  assert s.completed==1 and s.correct==1 and not s.fell,(q,sign,s.config)
print('Passed all 8 complete rotated goal-seeking routes',flush=True)
# Real observations preserve delayed reward exactly, with no orientation inputs.
e.observe=original_observe;o=e.reset();s=e.session;goal=s.goal_frame@np.array([1.18,1.18]);e.physics.data.qpos[:]=[goal[0],goal[1],0,0];e.physics.data.qvel[:]=0;e.physics.forward()
o=e.step(dict(reset=False,action=np.zeros(2)));assert o['reward']==1 and o['reward_received_previous'][0]==0
assert e.step(dict(reset=False,action=np.zeros(2)))['reward_received_previous'][0]==1
assert e.reset()['reward_received_previous'][0]==0
e.close()
report=dict(passed=True,frame_transition_cases=count,physics_routes=results,collision_orientations=4,delayed_reward_verified=True,existing_models=existing)
p=root/'artifacts/vc256-validation';p.mkdir(parents=True,exist_ok=True);(p/'results.json').write_text(json.dumps(report,indent=2))
print('ALL VC256 CHECKS PASSED',flush=True)
