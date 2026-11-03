"""Read-only compact inspection; safe while the single worker is active."""
import json
from pathlib import Path
import shutil
import sys
import site

site.addsitedir(str(Path.home() / 'AppData/Roaming/Python/Python313/site-packages'))
ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'results/g1_phase_supplement_20261031'
OUT = Path('E:/phasor_phase_supplement_20261031')


def read(path):
    return json.loads(path.read_text(encoding='utf8')) if path.exists() else None


def main():
    import psutil
    training = []
    for p in sorted((OUT / 'training').glob('*/manifest.json')):
        m = read(p); progress = read(p.parent / 'progress.json')
        training.append(dict(cell=p.parent.name, status=m['status'],
            updates=max(m.get('updates', 0), (progress or {}).get('updates', 0)),
            transitions=max(m.get('transitions', 0), (progress or {}).get('transitions', 0)),
            planned_updates=m['planned_updates']))
    rollouts = [read(p) for p in (OUT / 'rollouts').rglob('manifest.json')]
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_times', 'memory_info']):
        if (proc.info['name'] or '').lower() != 'python.exe':
            continue
        command = ' '.join(proc.info['cmdline'] or [])
        if 'g1_phase_supplement' in command:
            processes.append(dict(pid=proc.pid, command=command, cpu_seconds=proc.info['cpu_times'].user,
                                  memory_MiB=round(proc.info['memory_info'].rss / 1024 ** 2)))
    value = dict(queue=read(CONTROL / 'queue_status.json'), launch=read(CONTROL / 'launch.json'),
        train_cells=training, completed_train_cells=sum(r['status'] == 'COMPLETED' for r in training),
        train_transitions=sum(r['transitions'] for r in training),
        eval_groups=len(rollouts), completed_eval_groups=sum(r['status'] == 'COMPLETED' for r in rollouts),
        completed_eval_episodes=sum(r.get('episode_count', 0) for r in rollouts),
        free_disk_GiB={str(p): round(shutil.disk_usage(p).free / 1024 ** 3, 2) for p in (ROOT, Path('E:/'))},
        free_memory_GiB=round(psutil.virtual_memory().available / 1024 ** 3, 2), processes=processes,
        independent_audit=read(CONTROL / 'independent_audit.json'))
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
