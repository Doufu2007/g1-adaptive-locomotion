"""Execution-only two-worker scheduling; frozen scientific matrix unchanged."""
import argparse,concurrent.futures,gc,json,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from scripts import g1_phase_lr_factorial_20261027 as study
PREFIX="import site;site.addsitedir('C:/Users/郑傅韬/AppData/Roaming/Python/Python313/site-packages');import runpy,sys;"

def seal_check():
    seal=study.read(study.OUT/'parallel_execution_seal.json')
    for row in seal['files']:assert study.sha(row['path'])==row['sha256']
    for row in study.read(study.OUT/'extended_source_seal.json')['files']:assert study.sha(row['path'])==row['sha256']
    study.lock()

def cell(job):
    seal_check();study.train(job['arm'],job['lr_mode'],job['seed'])
    for point in study.CHECKPOINTS:
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        study.evaluate(job['arm'],job['lr_mode'],job['seed'],point)

def wait_inherited(handoff):
    job=handoff['job'];path=study.OUT/'training'/f"{job['arm']}_{job['lr_mode']}_{job['seed']}"/'manifest.json'
    while True:
        if path.exists():
            manifest=study.read(path)
            if manifest['status']=='COMPLETED':
                assert manifest['updates']==501 and manifest['transitions']==769536
                return
            assert manifest['status']=='RUNNING',manifest.get('error')
        try:
            process=psutil.Process(handoff['child_pid'])
            argv=' '.join(process.cmdline())
            assert 'g1_phase_lr_factorial_20261027' in argv and 'train' in argv,'Inherited PID identity changed'
        except psutil.NoSuchProcess:
            raise RuntimeError('Inherited training disappeared without completed checkpoint')
        time.sleep(10)

def worker(index,jobs,handoff):
    completed=[]
    status=study.OUT/f'parallel_worker_{index}.json'
    for job in jobs:
        name=f"{job['arm']}_{job['lr_mode']}_{job['seed']}"
        study.write(status,dict(status='RUNNING',at=study.now(),current=name,completed=completed))
        if handoff and job==handoff['job']:wait_inherited(handoff)
        log=study.OUT/'logs'/f'parallel_cell_{name}.log'
        args=['--stage','cell','--arm',job['arm'],'--lr-mode',job['lr_mode'],'--seed',str(job['seed'])]
        module='scripts.run_g1_phase_lr_factorial_parallel_20261027'
        command=PREFIX+'sys.argv='+repr([module]+args)+';runpy.run_module('+repr(module)+',run_name="__main__")'
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-u','-c',command],cwd=study.ROOT,
                env=dict(os.environ,PYTHONHASHSEED='20263300',PYTHONIOENCODING='utf-8'),stdout=stream,stderr=stream,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            study.write(status,dict(status='RUNNING',at=study.now(),current=name,child_pid=child.pid,completed=completed,log=str(log)))
            rc=child.wait()
        if rc:
            study.write(status,dict(status='FAILED',at=study.now(),current=name,completed=completed,exit_code=rc,log=str(log)))
            raise RuntimeError(f'{name} failed, exit={rc}')
        completed.append(name)
    study.write(status,dict(status='COMPLETED',at=study.now(),completed=completed))
    return completed

def queue():
    seal_check();assert study.read(study.OUT/'smoke_verification.json')['status']=='PASS'
    handoff=study.read(study.OUT/'controller_handoff.json');jobs=study.build_jobs()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i,jobs[i::2],handoff if i==0 else None) for i in range(2)]
        while not all(f.done() for f in futures):
            for f in futures:
                if f.done() and f.exception():
                    study.write(study.OUT/'queue_status.json',dict(status='FAILED',at=study.now(),error=repr(f.exception()),pid=os.getpid()))
                    raise f.exception()
            states=[]
            for i in range(2):
                p=study.OUT/f'parallel_worker_{i}.json'
                if p.exists():states.append(study.read(p))
            study.write(study.OUT/'queue_status.json',dict(status='RUNNING',at=study.now(),pid=os.getpid(),workers=states,
                maximum_concurrent_cells=2,scientific_matrix_unchanged=True))
            time.sleep(10)
        finished=[name for f in futures for name in f.result()]
    study.write(study.OUT/'queue_status.json',dict(status='TRAINING_AND_EVALUATION_COMPLETED',at=study.now(),pid=os.getpid(),completed=finished))
    module='scripts.analyze_g1_phase_lr_factorial_20261027'
    rc=subprocess.call([sys.executable,'-u','-c',PREFIX+'runpy.run_module('+repr(module)+',run_name="__main__")'],cwd=study.ROOT)
    if rc:raise RuntimeError('Final analysis failed: '+str(rc))
    study.write(study.OUT/'queue_status.json',dict(status='STAGE_A_COMPLETED',at=study.now(),pid=os.getpid(),completed=finished,
        analysis_sha256=study.sha(study.OUT/'analysis.json'),next_branch=study.read(study.OUT/'analysis.json')['candidate']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['queue','cell'],required=True)
    parser.add_argument('--arm');parser.add_argument('--lr-mode');parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.stage=='queue':queue()
    else:cell(dict(arm=args.arm,lr_mode=args.lr_mode,seed=args.seed,updates=501,transitions=769536))
