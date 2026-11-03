"""Serial, restart-aware execution of the frozen 12-cell phase/LR experiment."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from scripts import g1_phase_lr_factorial_20261027 as study
OUT=study.OUT
PREFIX="import site;site.addsitedir('C:/Users/郑傅韬/AppData/Roaming/Python/Python313/site-packages');import runpy,sys;"

def run(name,args,module='scripts.g1_phase_lr_factorial_20261027'):
    log=OUT/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
    argv=[sys.executable,'-u','-c',PREFIX+'sys.argv='+repr([module]+args)+';runpy.run_module('+repr(module)+',run_name="__main__")']
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen(argv,cwd=study.ROOT,env=dict(os.environ,PYTHONHASHSEED='20263300',PYTHONIOENCODING='utf-8'),
            stdout=stream,stderr=stream,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        study.write(OUT/'active_child.json',dict(name=name,pid=child.pid,started_at=study.now(),argv=argv,log=str(log)))
        rc=child.wait()
    with (OUT/'execution.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(name=name,exit_code=rc,ended_at=study.now(),log=str(log)))+'\n')
    if rc:raise RuntimeError(f'{name} failed with exit code {rc}; preserve raw attempt')

def main():
    completed=[]
    def work(name,args,module='scripts.g1_phase_lr_factorial_20261027'):
        study.write(OUT/'queue_status.json',dict(status='RUNNING',pid=os.getpid(),at=study.now(),current=name,completed=completed))
        run(name,args,module);completed.append(name)
    try:
        study.lock()
        if not (OUT/'instrumentation_parity.json').exists():work('replay_parity',['--stage','replay-check'])
        assert study.read(OUT/'instrumentation_parity.json')['status']=='PASS'
        for arm in ('clock','learned'):
            for lr in ('default','low'):
                stem=f'{arm}_{lr}_20261901'
                work('smoke_'+stem,['--stage','train','--arm',arm,'--lr-mode',lr,'--seed','20261901','--smoke'])
                work('smoke_eval_'+stem,['--stage','evaluate','--arm',arm,'--lr-mode',lr,'--seed','20261901','--updates','2','--smoke'])
        smoke=[study.read(OUT/'smoke'/f'{a}_{m}_20261901'/'manifest.json') for a in ('clock','learned') for m in ('default','low')]
        for field in ('initial_actor_sha256','initial_critic_sha256','common_cfg_sha256','common_agent_sha256','startup_model_hashes'):
            assert all(row[field]==smoke[0][field] for row in smoke),field
        assert all(row['updates']==2 and row['transitions']==3072 and row['encoder_frozen'] for row in smoke)
        study.write(OUT/'smoke_verification.json',dict(status='PASS',at=study.now(),four_groups=4,transitions=12288,
            equal_initial_actor_critic_task_physics=True,encoder_frozen=True))
        for job in study.build_jobs():
            arm,lr,seed=job['arm'],job['lr_mode'],job['seed'];stem=f'{arm}_{lr}_{seed}'
            common=['--arm',arm,'--lr-mode',lr,'--seed',str(seed)]
            work('train_'+stem,['--stage','train']+common)
            for point in study.CHECKPOINTS:
                work(f'eval_{stem}_u{point}',['--stage','evaluate']+common+['--updates',str(point)])
        study.write(OUT/'queue_status.json',dict(status='TRAINING_AND_EVALUATION_COMPLETED',pid=os.getpid(),at=study.now(),completed=completed))
    except BaseException as error:
        study.write(OUT/'queue_status.json',dict(status='FAILED',pid=os.getpid(),at=study.now(),completed=completed,error=repr(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
