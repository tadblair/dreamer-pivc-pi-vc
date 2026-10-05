"""Collision-resolved elevated track with online session rules and FiLM observations."""
import functools
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import elements
import embodied
from dm_control import mujoco
from shapely.geometry import Polygon
import shapely
from elevated_track_barriers import add_barriers, set_barrier_configuration
from elevated_track_screens import add_screens
from elevated_session import Session, CORNERS
from ray_sensors import cast_rays, ray_directions

ROOT = Path(__file__).resolve().parents[1]
RADIUS = .0185
DT = .002
SUBSTEPS = 125
MAX_SPEED = .30
MAX_TURN = np.pi
OUTLINE = np.array([[-.95,1.06],[.95,1.06],[1.1875,1.3],[1.3,1.1875],[1.06,.95],[1.06,-.95],
 [1.3,-1.1875],[1.1875,-1.3],[.95,-1.06],[-.95,-1.06],[-1.1875,-1.3],[-1.3,-1.1875],
 [-1.06,-.95],[-1.06,.95],[-1.3,1.1875],[-1.1875,1.3]])
HOLE = np.array([[-.94,.94],[-.06,.94],[-.06,.06],[-.94,.06]])


class ElevatedTrack(embodied.Env):
    def __init__(self, task='session', seed=0, size=64, logdir=None, corner_discs=True, always_dim_east=False, delayed_reward_input=False):
        assert task == 'session' and size == 64
        self.delayed_reward_input = bool(delayed_reward_input)
        self._previous_reward_received = np.zeros(1, np.float32)
        self.rng = np.random.RandomState(seed)
        self.seed = int(seed)
        self.session_number = 0
        self.logdir = Path(str(logdir)) if logdir else None
        if self.logdir: self.logdir.mkdir(parents=True, exist_ok=True)
        tree = ET.parse(ROOT/'artifacts/elevated-track/scene.xml')
        self.barriers = add_barriers(tree, 0, dynamic=True)
        add_screens(tree)
        self.always_dim_east = bool(always_dim_east)
        if self.always_dim_east:
            tree.find("asset/material[@name='screen_east']").set("rgba", f"{50/(255*1000)} {50/(255*1000)} {50/(255*1000)} 1")
        world = tree.find('worldbody')
        self.corner_discs = bool(corner_discs)
        if self.corner_discs:
            ET.SubElement(tree.find('asset'), 'material', name='corner_silver',
                          rgba='.75 .75 .75 1', specular='.9', shininess='.8')
            for name, (x, y) in zip(['NE', 'NW', 'SE', 'SW'], CORNERS):
                ET.SubElement(world, 'geom', name='corner_disc_'+name, type='cylinder',
                              pos=f'{x} {y} 1.005', size='.02 .005', material='corner_silver',
                              contype='0', conaffinity='0')
        world.remove(world.find("body[@name='camera_rig']"))
        body = ET.SubElement(world, 'body', name='agent', pos=f'0 0 {1+RADIUS}')
        # Upright body, four physical DOFs; no kinematic position updates during steps.
        for name, axis, kind in [('x','1 0 0','slide'),('y','0 1 0','slide'),
                                 ('z','0 0 1','slide'),('yaw','0 0 1','hinge')]:
            ET.SubElement(body,'joint',name='agent_'+name,type=kind,axis=axis,damping='0')
        ET.SubElement(body,'inertial',pos='0 0 0',mass='.03',diaginertia='.00001 .00001 .00001')
        ET.SubElement(body,'geom',name='agent_body',type='sphere',size=str(RADIUS),
                      rgba='.12 .12 .12 1',friction='.25 .001 .0001',solref='.004 1',solimp='.99 .999 .0001')
        for name,side,yaw in [('left_eye',1,50),('right_eye',-1,-50)]:
            a,p=np.deg2rad([yaw,15])
            forward=np.array([np.cos(a)*np.cos(p),np.sin(a)*np.cos(p),np.sin(p)])
            right=np.array([np.sin(a),-np.cos(a),0.]);up=np.cross(right,forward)
            ET.SubElement(body,'camera',name=name,pos=f'0 {side*.0065} {.07-RADIUS}',
                          xyaxes=' '.join(map(str,np.r_[right,up])),fovy='150')
        # Stable small-step contact integration. Keep physical track/rim/barrier geoms.
        tree.find('option').set('timestep',str(DT))
        tree.find('option').set('integrator','implicitfast')
        tree.find('option').set('iterations','50')
        self.scene_xml=ET.tostring(tree.getroot(),encoding='unicode')
        self.physics=mujoco.Physics.from_xml_string(self.scene_xml)
        self.body_id=self.physics.model.name2id('agent','body')
        self.geom_id=self.physics.model.name2id('agent_body','geom')
        self.screen_id=self.physics.model.name2id('screen_east','material')
        self.support=Polygon(OUTLINE,holes=[HOLE+s for s in ([0,0],[1,0],[0,-1],[1,-1])]).buffer(.02,join_style='mitre')
        shapely.prepare(self.support)
        self.session=None
        self._done=True
        self._closed=False

    @functools.cached_property
    def obs_space(self):
        return dict(**({'reward_received_previous': elements.Space(np.float32, (1,), 0, 1)} if self.delayed_reward_input else {}), image=elements.Space(np.uint8,(64,64,2)),
            source_image=elements.Space(np.uint8,(64,64,2)),
            vibrissal=elements.Space(np.float32,(61,),0,1),
            efference=elements.Space(np.float32,(2,),-1,1),
            motion=elements.Space(np.float32,(3,)),reward=elements.Space(np.float32),
            is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool),
            **{f'log/{k}':elements.Space(np.float32) for k in
               ['trials_completed','choices','correct_choices','accuracy_percent','fell','session_steps','config','no_choice_trials','prox_choices','prox_correct','dist_choices','dist_correct','session_number']})

    @functools.cached_property
    def act_space(self):
        return dict(action=elements.Space(np.float32,(2,),-1,1),reset=elements.Space(bool))

    def pose(self):
        return self.physics.data.xpos[self.body_id].copy(), float(self.physics.data.qpos[3])

    def sync(self):
        if self.barriers['configuration'] != self.session.config:
            set_barrier_configuration(self.physics,self.barriers,self.session.config)
        value=(255 if self.session.available and not self.always_dim_east else 50)/(255*1000)
        self.physics.model.mat_rgba[self.screen_id,:3]=value

    def supported(self, position):
        # An upright body is unsupported once its center projection leaves track+rim.
        return bool(shapely.intersects_xy(self.support,position[0],position[1]))

    def reset(self):
        self._previous_reward_received = np.zeros(1, np.float32)
        self.physics.reset()
        self.session=Session(self.rng)
        self.session_number += 1
        prox=self.session.sequence[0]==1
        self.physics.data.qpos[:]=[0,.25 if prox else -.25,0,-np.pi/2 if prox else np.pi/2]
        self.physics.data.qvel[:]=0
        self.physics.data.qfrc_applied[:]=0
        self.physics.forward()
        # Restore all panels after physics.reset() resets mocap positions.
        set_barrier_configuration(self.physics,self.barriers,0)
        self.sync()
        self._done=False
        if self.logdir:
            with (self.logdir/'session_starts.jsonl').open('a') as f:
                f.write(json.dumps(dict(session=self.session_number,seed=self.seed,
                    sequence=['prox' if x==1 else 'dist' for x in self.session.sequence]))+'\n')
        return self.observe(0.,True,np.zeros(2,np.float32),None)

    def step(self, action):
        if action['reset'] or self._done:return self.reset()
        command=np.clip(np.asarray(action['action'],np.float64),-1,1)
        before=self.pose()
        self.session.begin_step()
        reward=0.
        for k in range(SUBSTEPS):
            angle=self.physics.data.qpos[3]
            desired=command[0]*MAX_SPEED*np.array([np.cos(angle),np.sin(angle)])
            # Force-limited velocity servos: contact solver, not teleportation, resolves walls.
            force=2.*(desired-self.physics.data.qvel[:2])
            self.physics.data.qfrc_applied[:2]=np.clip(force,-.6,.6)
            self.physics.data.qfrc_applied[3]=np.clip(.001*(command[1]*MAX_TURN-self.physics.data.qvel[3]),-.005,.005)
            self.physics.step()
            pos,_=self.pose()
            reward+=self.session.sample(pos[:2],self.supported(pos),fraction=(k+1)/SUBSTEPS,body_z=pos[2])
            self.sync()
            if self.session.done:break
        self.session.end_step()
        self._done=self.session.done
        obs=self.observe(reward,False,command.astype(np.float32),before)
        # The observation at t contains b_(t-1), never the current reward b_t.
        self._previous_reward_received = np.array([reward > 0], np.float32)
        if self._done and self.logdir:
            with (self.logdir/'sessions.jsonl').open('a') as f:
                f.write(json.dumps(dict(session=self.session_number,steps=self.session.steps,
                    trials_completed=self.session.completed,choices=self.session.choices,
                    correct=self.session.correct,accuracy_percent=self.session.accuracy,
                    fell=self.session.fell,terminal=self.session.terminal,
                    trial_choices=self.session.trial_choices,transitions=self.session.events))+'\n')
        return obs

    def observe(self,reward,first,command,before):
        pos,angle=self.pose()
        images=[]
        for camera in ['left_eye','right_eye']:
            rgb=self.physics.render(64,64,camera_id=camera,render_flag_overrides={'shadow':False})
            images.append(np.rint(rgb.astype(np.float32)@np.array([.299,.587,.114],np.float32)).astype(np.uint8))
        image=np.stack(images,-1)
        dirs=ray_directions([np.cos(angle),np.sin(angle)])
        dist,_,_=cast_rays(self.physics.model.ptr,self.physics.data.ptr,pos,dirs,[self.geom_id])
        tactile=np.where(dist>=0,np.clip((.111-dist)/(.111-RADIUS),0,1),0).astype(np.float32)
        if first:
            source=image.copy();source_tactile=tactile.copy();motion=np.zeros(3,np.float32)
        else:
            source=self._image;source_tactile=self._tactile
            previous,heading=before;delta=pos[:2]-previous[:2]
            motion=np.array([(delta@[np.cos(heading),np.sin(heading)])/RADIUS,
                (delta@[-np.sin(heading),np.cos(heading)])/RADIUS,
                np.arctan2(np.sin(angle-heading),np.cos(angle-heading))/(np.pi/6)],np.float32)
        self._image=image.copy();self._tactile=tactile.copy()
        s=self.session
        by_start = {}
        for label, value in [('prox', 1), ('dist', -1)]:
            choices = [c for c in s.trial_choices if s.sequence[c['trial']-1] == value]
            by_start[label+'_choices'] = len(choices)
            by_start[label+'_correct'] = sum(c['corner'] == 0 for c in choices)
        metrics=dict(**by_start,session_number=self.session_number,trials_completed=s.completed,choices=s.choices,correct_choices=s.correct,
            accuracy_percent=s.accuracy,fell=s.fell,session_steps=s.steps,config=s.config,
            no_choice_trials=int(s.choice is None and s.completed<32))
        return dict(**({'reward_received_previous': self._previous_reward_received.copy()} if self.delayed_reward_input else {}), image=image,source_image=source,vibrissal=source_tactile,efference=command,
            motion=motion,reward=np.float32(reward),is_first=bool(first),is_last=bool(s.done),
            is_terminal=bool(s.terminal),**{f'log/{k}':np.float32(v) for k,v in metrics.items()})

    def close(self):
        if not self._closed:
            self.physics.free();self._closed=True
