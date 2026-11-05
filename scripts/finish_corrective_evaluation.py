from pathlib import Path
import sys,time,json,subprocess,os
B=Path(__file__).resolve().parents[1];R=B/'training_completion'
def done(b):
    p=R/('corrective_'+b+'_status.json');return p.exists() and json.loads(p.read_text()).get('state')=='complete'
def main():
    grace=time.time()+30
    while not all(done(b) for b in ('donor','pooled')):
        for b in ('donor','pooled'):
            if time.time()>grace and not done(b) and not (R/('corrective_'+b+'.lock')).exists():raise RuntimeError(b+' corrective worker stopped before completion')
        time.sleep(5)
    for key,script in [('calibration','calibrate_training_completion.py'),('auxiliary','evaluate_completion_auxiliary.py'),('audit','analyse_training_completion.py')]:
        with (R/(key+'.stdout.log')).open('ab') as out,(R/(key+'.stderr.log')).open('ab') as err:
            p=subprocess.Popen([sys.executable,'-B','-u',str(B/'scripts'/script)],cwd=B,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);(R/(key+'.pid')).write_text(str(p.pid));code=p.wait()
        if code:raise RuntimeError(key+' failed')
    assert json.loads((R/'audit.json').read_text())['state']=='all_registered_additional_runs_complete'
    (R/'evaluation_status.json').write_text(json.dumps({'state':'complete','updated_epoch':time.time()}))
if __name__=='__main__':
    lock=R/'evaluation_manager.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
