"""Single-instance, auditable handoff from verified download to real analysis."""
from pathlib import Path
import os,sys,time,json,subprocess,traceback
R=Path(__file__).resolve().parent;LOCK=R/'pipeline.lock'
def state(s,**kw):
 p=R/'pipeline_status.tmp';p.write_text(json.dumps({'state':s,'pid':os.getpid(),'updated_epoch':time.time(),**kw},indent=2),encoding='utf-8');os.replace(p,R/'pipeline_status.json')
def execute(name):
 state('running',script=name)
 env=dict(os.environ,PYTHONUTF8='1')
 with (R/(name+'.stdout.log')).open('w',encoding='utf-8') as out,(R/(name+'.stderr.log')).open('w',encoding='utf-8') as err:
  p=subprocess.run([sys.executable,'-B',str(R/name)],cwd=str(R.parent.parent.parent.parent),env=env,stdout=out,stderr=err)
 if p.returncode:raise RuntimeError(f'{name} failed; see persistent stderr log')
if __name__=='__main__':
 fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  while True:
   st=json.loads((R/'download_status.json').read_text())
   if st['state']=='failed':raise RuntimeError('Download reported failure; leave its resumable chunks intact')
   if st['state'] in ['download_complete_pending_gzip_crc','complete']:break
   state('waiting_for_download',verified_bytes=st['verified_bytes'],expected_bytes=st['expected_bytes']);time.sleep(20)
  if not (R/'prepared/count_provenance.json').exists():execute('prepare_external.py')
  if not (R/'results/status.json').exists() or json.loads((R/'results/status.json').read_text())['stage']!='complete':execute('evaluate_frozen.py')
  state('analysis_complete_requires_manuscript_review',result='results/summary.json',paper_automatically_changed=False)
 except BaseException as e:state('failed',error=str(e),traceback=traceback.format_exc());raise
 finally:LOCK.unlink(missing_ok=True)
