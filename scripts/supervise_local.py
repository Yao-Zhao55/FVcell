"""Local continuation: watch download completion and launch one training run.

No cloud, OS scheduler, purchased resource or original input modification.
The app heartbeat handles notifications, diagnosis and stopped workers.
"""
from pathlib import Path
import json,os,sys,time,subprocess,datetime
from atomic_json import write_json
BASE=Path(__file__).resolve().parents[1]
def save(obj):
    obj['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    write_json(BASE/'supervisor_status.json',obj)
def main():
    lock=BASE/'supervisor.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        while True:
            status_path=BASE/'download_status.json'
            try:status=json.loads(status_path.read_text(encoding='utf-8'))
            except (OSError,json.JSONDecodeError):time.sleep(15);continue
            if status.get('state')=='complete':
                if (BASE/'training_first/results.json').exists():
                    save({'state':'first_training_complete','pid':os.getpid()});return
                if (BASE/'training_first/training.lock').exists():
                    save({'state':'existing_training_lock_needs_process_check','pid':os.getpid()});return
                save({'state':'launching_validation_and_training','pid':os.getpid()})
                with (BASE/'training.stdout.log').open('a',encoding='utf-8') as out,(BASE/'training.stderr.log').open('a',encoding='utf-8') as err:
                    process=subprocess.Popen([sys.executable,'-B','-u',str(BASE/'scripts/run_first_training.py')],cwd=BASE,stdout=out,stderr=err)
                    (BASE/'training_process_id.txt').write_text(str(process.pid),encoding='utf-8')
                    save({'state':'validation_or_training_running','pid':os.getpid(),'training_pid':process.pid})
                    code=process.wait()
                save({'state':'first_training_complete' if code==0 else 'training_blocked_or_failed','exit_code':code,'pid':os.getpid()});return
            save({'state':'waiting_for_verified_download','pid':os.getpid(),'download_state':status.get('state'),'asset':status.get('asset'),'verified_bytes':status.get('verified_downloaded_bytes'),'download_updated_epoch':status.get('updated_epoch')})
            time.sleep(15)
    except BaseException as error:
        save({'state':'supervisor_failed','pid':os.getpid(),'error':str(error)});raise
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
