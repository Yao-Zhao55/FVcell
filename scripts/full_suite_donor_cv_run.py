from full_suite_core import *
from full_suite_run import freeze_plan,run_task
import argparse
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start-fold',type=int,default=0);parser.add_argument('--end-fold',type=int,default=4);args=parser.parse_args()
    if not 0<=args.start_fold<args.end_fold<=4:raise ValueError('Invalid fold range')
    p=json.loads((OUT/'E1_donor_crossvalidation_protocol.json').read_text());order=p['donor_order'];protocol=freeze_plan()
    y=np.load(PREP/'donor_effects.npy',mmap_mode='r');meta=pd.read_csv(PREP/'donor_rows.csv');g=pd.read_csv(PREP/'donor_genes.csv').gene_id.to_numpy()
    meta['selection_or_calibration']=['calibration' if hashnum('calibration|'+str(t))%2 else 'selection' for t in meta.target_contrast]
    for i in range(args.start_fold,args.end_fold):
        m=meta.copy();m['partition']='train';m.loc[m.donor_id==order[i],'partition']='test';m.loc[m.donor_id==order[(i+1)%4],'partition']='validation'
        task='E1_donor_P1' if i==0 else 'E1_donor_P1_fold'+str(i)
        run_task(task,y,m,g,p['models_P1'],protocol,'P1')
    complete=[]
    for i in range(4):
        task='E1_donor_P1' if i==0 else 'E1_donor_P1_fold'+str(i)
        if all((OUT/'runs'/task/f/'summary.json').exists() for f in p['models_P1']):complete.append(i)
    write_json(OUT/'donor_cv_status.json',{'state':'complete' if len(complete)==4 else 'partial','completed_folds':complete,'outer_test_donors':4,'protocol_sha256':sha(OUT/'E1_donor_crossvalidation_protocol.json'),'updated_epoch':time.time()})
if __name__=='__main__':
    lock=OUT/'donor_cv.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
