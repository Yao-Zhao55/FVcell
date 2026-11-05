"""Continue this local job after all training workers finish; no scheduler."""
from pathlib import Path
import json,os,time,subprocess,sys
B=Path(__file__).resolve().parents[1];R=B/'training_completion'
def done(name):
    p=R/(name+'_status.json');return p.exists() and json.loads(p.read_text()).get('state')=='complete'
def main():
    while not all(done(n) for n in ('hierarchy','context_state','missing')):
        if not done('hierarchy') and not (R/'hierarchy.lock').exists():raise RuntimeError('Hierarchy worker ended without success receipt')
        if not all(done(n) for n in ('context_state','missing')) and not (R/'queue.lock').exists():raise RuntimeError('Queued training ended without success receipts')
        time.sleep(5)
    for key,script in [('calibration','calibrate_training_completion.py'),('audit','analyse_training_completion.py')]:
        with (R/(key+'.stdout.log')).open('ab') as out,(R/(key+'.stderr.log')).open('ab') as err:
            p=subprocess.Popen([sys.executable,'-B','-u',str(B/'scripts'/script)],cwd=B,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);(R/(key+'.pid')).write_text(str(p.pid));code=p.wait()
        if code:raise RuntimeError(key+' failed; see error log')
    (R/'evaluation_status.json').write_text(json.dumps({'state':'complete','updated_epoch':time.time()}))
if __name__=='__main__':
    lock=R/'evaluation_manager.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
