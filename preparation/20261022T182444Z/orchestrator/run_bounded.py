"""Run one authorized local command with a process time limit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent.parent

def owned(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PACKAGE):
        raise ValueError('Keep the command and its receipt under this preparation package.')
    return path

def run(script, args, receipt_dir, max_seconds):
    script, receipt_dir = owned(script), owned(receipt_dir)
    if not script.is_file():
        raise FileNotFoundError(script)
    if receipt_dir.exists():
        raise FileExistsError('Use a new receipt directory for each attempt.')
    receipt_dir.mkdir(parents=True)
    cache = PACKAGE/'execution_cache'
    env = os.environ.copy()
    for key, name in [('TEMP','tmp'),('TMP','tmp'),('PIP_CACHE_DIR','pip'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('WARP_CACHE_PATH','warp')]:
        path = cache/name
        path.mkdir(parents=True,exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [sys.executable, str(script), *args]
    receipt = {'status':'RUNNING','command':command,'max_seconds':max_seconds,
               'preparation_package':str(PACKAGE),'timeout':False}
    receipt_path = receipt_dir/'process_receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    started = time.monotonic()
    with (receipt_dir/'stdout.log').open('w') as stdout, (receipt_dir/'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command,cwd=PACKAGE,env=env,stdin=subprocess.DEVNULL,
                                   stdout=stdout,stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        receipt['pid'] = process.pid
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        try:
            code = process.wait(timeout=max_seconds)
        except subprocess.TimeoutExpired:
            receipt['timeout'] = True
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,check=False)
            else:
                process.kill()
            process.wait(timeout=30)
            code = 124
    receipt.update(status='PASS' if code==0 else 'BLOCKED',exit_code=code,
                   elapsed_seconds=time.monotonic()-started)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return code

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script',type=Path,required=True)
    parser.add_argument('--receipt-dir',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 14400:
        parser.error('Use a time limit above 0 and at most 14400 seconds.')
    arguments = args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    return run(args.script,arguments,args.receipt_dir,args.max_seconds)

if __name__=='__main__':
    raise SystemExit(main())
