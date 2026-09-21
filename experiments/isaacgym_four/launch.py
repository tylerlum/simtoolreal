"""Launch the selected resolved runs in persistent tmux sessions on idle GPUs."""
import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--python', default=sys.executable, help='Python 3.8 with the Gym dependencies')
    parser.add_argument('--dry-run', action='store_true', help='Show commands and occupancy without starting jobs')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not manifest or len({m['gpu'] for m in manifest}) != len(manifest):
        parser.error('Expected at least one run and one distinct GPU per run')
    gpus = {int(r[0].strip()): r[1].strip() for r in csv.reader(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
    occupied = {r[0].strip() for r in csv.reader(subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines())}
    plans = []
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for m in manifest:
        if m['gpu'] not in gpus:
            parser.error('GPU does not exist: ' + str(m['gpu']))
        if not Path(m['config']).is_file():
            parser.error('Missing resolved config: ' + m['config'])
        if Path(m['run_dir']).exists():
            parser.error('Run directory already exists: ' + m['run_dir'])
        uuid = gpus[m['gpu']]
        command = ['env', 'CUDA_VISIBLE_DEVICES=' + uuid, 'STR_PYTHON=' + args.python,
                   str(ROOT / 'run_worker.sh'), m['config']]
        plans.append((m, uuid, command))
        if args.dry_run:
            print(json.dumps(dict(key=m['key'], gpu=m['gpu'], occupied=uuid in occupied,
                                  command=shlex.join(command))))
        elif uuid in occupied:
            parser.error('GPU {} occupied; refusing launch'.format(m['gpu']))
    if args.dry_run:
        return
    if shutil.which('tmux') is None:
        parser.error('tmux is required')
    version = subprocess.check_output([args.python, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], text=True).strip()
    if version != '3.8':
        parser.error('This tested Isaac Gym recipe requires Python 3.8; got ' + version)
    for m, uuid, command in plans:
        run_dir = Path(m['run_dir'])
        run_dir.mkdir(parents=True)
        m.update(session='strgym_' + timestamp + '_' + m['key'], gpu_uuid=uuid,
                 launched_utc=datetime.now(timezone.utc).isoformat())
        script = run_dir / 'start.sh'
        script.write_text('#!/usr/bin/env bash\nset +e\n' + shlex.join(command)
                          + ' > ' + shlex.quote(str(run_dir / 'worker.log')) + ' 2>&1\n'
                          + 'status=$?\nprintf "%s\\n" "$status" > '
                          + shlex.quote(str(run_dir / 'supervisor_exit_status.txt')) + '\nexit "$status"\n')
        script.chmod(0o755)
        (run_dir / 'launch.json').write_text(json.dumps(m, indent=2) + '\n')
        subprocess.run(['tmux', 'new-session', '-d', '-s', m['session'], 'bash', str(script)], check=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
        print(m['key'], m['gpu'], m['session'], m['wandb_url'], flush=True)


if __name__ == '__main__':
    main()
