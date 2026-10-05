"""VC256 session: independently rotated goal frames and canonical return frames."""
import numpy as np
from elevated_session import Session as FixedSession
from elevated_track_barriers import BARRIERS, CONFIGS
ROTATIONS = np.array([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]]])
REFLECT = np.diag([-1,1])

def return_frame(goal, arm):
    near = np.dot(goal, arm) > 0
    canonical_arm = np.array([0,1 if near else -1])
    rotation = next(q for q in ROTATIONS if np.array_equal(q @ canonical_arm, arm))
    local_goal = rotation.T @ goal
    transform = rotation @ (REFLECT if local_goal[0] < 0 else np.eye(2,dtype=int))
    assert np.array_equal(transform @ np.array([1,1]), goal)
    return (3 if near else 4), transform

def raised_doors(config, transform):
    result=set()
    for number in CONFIGS[config]:
        x,y,orientation=BARRIERS[f'B{number}']
        target=transform @ [x,y]
        names=[name for name,(bx,by,_) in BARRIERS.items() if np.allclose(target,[bx,by],atol=1e-10)]
        assert len(names)==1
        result.add(names[0])
    return result

class Session(FixedSession):
    def __init__(self,rng,sequence=None):
        super().__init__(rng,sequence)
        self.rng=rng
        self.orientation=int(rng.randint(4))
        self.goal_frame=ROTATIONS[self.orientation].copy()
        self.return_frame=self.goal_frame.copy()
        self.frame_history=[dict(trial=1,orientation=self.orientation)]

    @property
    def door_frame(self):
        return self.goal_frame if self.config in (0,1,2) else self.return_frame

    def sample(self,xy,supported=True,fraction=1.,body_z=1.0185):
        world=np.asarray(xy)
        old_count=self.completed
        old_goal=self.goal_frame @ np.array([1,1])
        frame=self.goal_frame if self.available else self.return_frame
        reward=super().sample(world @ frame,supported,fraction,body_z)
        if self.completed > old_count and not self.done:
            self.orientation=int(self.rng.randint(4))
            self.goal_frame=ROTATIONS[self.orientation].copy()
            arm=self.goal_frame @ np.array([0,int(self.sequence[self.completed])])
            self.config,self.return_frame=return_frame(old_goal,arm)
            self.events[-1]['new']=self.config
            self.events[-1].update(next_orientation=self.orientation,return_transform=self.return_frame.tolist(),start_arm=arm.tolist())
            self.previous_y=float((world @ self.return_frame)[1])
            self.frame_history.append(dict(trial=self.completed+1,orientation=self.orientation,start_arm=arm.tolist()))
        return reward