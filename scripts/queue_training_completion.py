"""Serialize extra branches after the context worker, avoiding duplicate tasks."""
from completion_core import *
import subprocess
def main():
    while True:
        p=ROOT/'context_status.json'
        if p.exists() and json.loads(p.read_text()).get('state')=='complete':break
        if not (ROOT/'context.lock').exists():raise RuntimeError('Context worker ended before a completion receipt')
        time.sleep(5)
    for branch in ('context_state','missing'):
        status=ROOT/(branch+'_status.json')
        if status.exists() and json.loads(status.read_text()).get('state')=='complete':continue
        assert not (ROOT/(branch+'.lock')).exists(),'Branch already running'
        with (ROOT/(branch+'.stdout.log')).open('ab') as out,(ROOT/(branch+'.stderr.log')).open('ab') as err:
            p=subprocess.Popen([sys.executable,'-B','-u',str(BASE/'scripts/run_training_completion.py'),branch],cwd=BASE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            (ROOT/(branch+'.pid')).write_text(str(p.pid));code=p.wait()
        if code:raise RuntimeError(branch+' failed; inspect its error log')
    write_json(ROOT/'queue_status.json',{'state':'complete','updated_epoch':time.time()})
if __name__=='__main__':
    lock=ROOT/'queue.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
