from full_suite_core import *
from full_suite_run import load_primary,freeze_plan,run_task
def main():
    y,m,g=load_primary();scope=json.loads((OUT/'functional_scope_lock.json').read_text());targetset=set(scope['mapped_target_ids'])
    m['partition']=m.global_target_partition;m.loc[(m.partition=='test')&~m.target_contrast.isin(targetset),'partition']='excluded';m.loc[m.target_contrast.isin(targetset),'partition']='test'
    assert not set(m.loc[m.partition=='train','target_contrast'])&set(m.loc[m.partition.isin(['validation','test']),'target_contrast'])
    run_task('E5_functional_targets_strict_P1',y,m,g,['ridge','concat','fv'],freeze_plan(),'P1')
    write_json(OUT/'functional_strict_status.json',{'state':'complete','protocol_sha256':sha(OUT/'E5_strict_validation_protocol.json'),'updated_epoch':time.time()})
if __name__=='__main__':
    lock=OUT/'functional_strict.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
