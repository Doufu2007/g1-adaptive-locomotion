"""UTF-8 native launcher, no PowerShell profile dependency."""
import os
from pathlib import Path
import site
import sys

site.addsitedir(str(Path.home() / 'AppData/Roaming/Python/Python313/site-packages'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'resources':
        import json
        import psutil
        print(json.dumps(dict(memory=psutil.virtual_memory()._asdict(),
            processes=[dict(pid=p.pid, name=p.name(), cpu=p.cpu_times().user,
                            cmd=p.cmdline(), rss=p.memory_info().rss)
                       for p in psutil.process_iter() if p.name().lower() in ('python.exe', 'powershell.exe')]),
                         ensure_ascii=False, indent=2))
    else:
        from scripts.g1_phase_supplement_20261031 import main
        try:
            main()
        except BaseException as error:
            from scripts.g1_phase_supplement_20261031 import failure_evidence
            try:
                failure_evidence(error)
            except Exception:
                import traceback
                traceback.print_exc()
            raise
