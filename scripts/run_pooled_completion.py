"""Joint model checks, registered before reading any new test summaries."""
from completion_core import *
from full_suite_run import load_primary
def main():
    while True:
        p=ROOT/'missing_status.json'
        if p.exists() and json.loads(p.read_text()).get('state')=='complete':break
        if not (ROOT/'queue.lock').exists():raise RuntimeError('Missingness queue ended without success')
        time.sleep(5)
    os.environ['COMPLETION_BRANCH']='pooled';y,m,g=load_primary()
    for name,source,permission in [('hierarchical_combo_P2','E1_P2_combo','P2'),('hierarchical_global_P1','E1_global_target_P1','P1'),('hierarchical_functional_P1','E5_functional_targets_strict_P1','P1')]:
        mm=pd.read_csv(OUT/'runs'/source/'rows.csv');run(name,y,mm,g,['hierarchical_pooled'],permission)
    mm=m.copy();mm['partition']=mm.global_target_partition
    for n in (70,30):
        mask=make_mask(y,mm,n/100);run('missing_'+str(n)+'_global_P1',y,mm,g,['hierarchical_pooled'],mask=mask);del mask;gc.collect()
    write_json(ROOT/'pooled_status.json',{'state':'complete','updated_epoch':time.time()})
if __name__=='__main__':
    lock=ROOT/'pooled.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
