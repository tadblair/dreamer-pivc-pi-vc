"""Single balanced-session criterion, with checkpointed first-reward timing."""
import json
from pathlib import Path
class BalancedSessionCriterion:
    def __init__(self, logdir):
        self.root=Path(str(logdir));self.root.mkdir(parents=True,exist_ok=True)
        self.first_reward=None;self.success=None;self.solved=False
    def _write(self,name,data):
        p=self.root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
    def observe(self,tran,worker,step):
        if self.solved:return
        step=int(step)
        if self.first_reward is None and not tran['is_first'] and int(tran['log/trials_completed'])>0:
            self.first_reward=dict(global_step=step,worker=int(worker),session=int(tran['log/session_number']),session_step=int(tran['log/session_steps']))
            self._write('first_reward.json',self.first_reward)
        if not tran['is_last']:return
        row=dict(global_step=step,worker=int(worker),session=int(tran['log/session_number']),session_steps=int(tran['log/session_steps']),rewards=int(tran['log/trials_completed']),fell=bool(tran['log/fell']))
        for key in ['prox_choices','prox_correct','dist_choices','dist_correct']:row[key]=int(tran['log/'+key])
        row['criterion_met']=(row['rewards']==32 and not row['fell'] and row['prox_choices']==16 and row['dist_choices']==16 and row['prox_correct']>=12 and row['dist_correct']>=12)
        with (self.root/'balanced_sessions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if row['criterion_met']:
            assert self.first_reward is not None
            self.success=dict(**row,first_reward=self.first_reward,steps_since_first_reward=step-self.first_reward['global_step'])
            self.solved=True
    def save(self):
        return dict(first_reward=self.first_reward,success=self.success,solved=self.solved)
    def load(self,data):
        self.first_reward=data['first_reward'];self.success=data['success'];self.solved=data['solved']
        if self.first_reward:self._write('first_reward.json',self.first_reward)
    def finish(self,step):
        if self.solved:
            assert int(step)==self.success['global_step']
            cp=(self.root/'ckpt/latest').read_text().strip()
            self._write('criterion_reached.json',dict(**self.success,checkpoint=cp,checkpoint_verified=True))
