"""Finish only the missing official baseline in donor fold 0.

The other CV worker handles folds 1-3 only. No task-family overlaps.
"""
from full_suite_core import *
from full_suite_run import freeze_plan,run_task

def main():
    p=json.loads((OUT/'E1_donor_crossvalidation_protocol.json').read_text());order=p['donor_order']
    y=np.load(PREP/'donor_effects.npy',mmap_mode='r');m=pd.read_csv(PREP/'donor_rows.csv');g=pd.read_csv(PREP/'donor_genes.csv').gene_id.to_numpy()
    m['partition']='train';m.loc[m.donor_id==order[0],'partition']='test';m.loc[m.donor_id==order[1],'partition']='validation'
    m['selection_or_calibration']=['calibration' if hashnum('calibration|'+str(t))%2 else 'selection' for t in m.target_contrast]
    run_task('E1_donor_P1',y,m,g,['state_official'],freeze_plan(),'P1')
    write_json(OUT/'runs/E1_donor_P1/complete.json',{'state':'complete','families':p['models_P1'],'updated_epoch':time.time()})
    complete=[]
    for i in range(4):
        t='E1_donor_P1' if i==0 else 'E1_donor_P1_fold'+str(i)
        if all((OUT/'runs'/t/f/'summary.json').exists() for f in p['models_P1']):complete.append(i)
    write_json(OUT/'donor_cv_status.json',{'state':'complete' if len(complete)==4 else 'partial','completed_folds':complete,'outer_test_donors':4,'protocol_sha256':sha(OUT/'E1_donor_crossvalidation_protocol.json'),'updated_epoch':time.time()})
if __name__=='__main__':
    lock=OUT/'donor0_state.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
