from completion_core import *
from full_suite_run import load_primary
import argparse

def main(branch):
    freeze();os.environ['COMPLETION_BRANCH']=branch
    if branch=='hierarchy':
        y=np.load(PREP/'donor_effects.npy',mmap_mode='r');m=pd.read_csv(PREP/'donor_rows.csv');g=pd.read_csv(PREP/'donor_genes.csv').gene_id.to_numpy();order=['CE0008678','CE0010866','CE0008162','CE0006864']
        m['selection_or_calibration']=['calibration' if hashnum('calibration|'+t)%2 else 'selection' for t in m.target_contrast]
        for i in range(4):
            mm=m.copy();mm['partition']='train';mm.loc[mm.donor_id==order[i],'partition']='test';mm.loc[mm.donor_id==order[(i+1)%4],'partition']='validation'
            run('hierarchical_donor_fold'+str(i),y,mm,g,['fixed_gaussian','hierarchical','unshrunk_offsets'])
    elif branch=='missing':
        y,m,g=load_primary();m['partition']=m.global_target_partition
        for rate in (.7,.3):
            mask=make_mask(y,m,rate);run('missing_'+str(int(rate*100))+'_global_P1',y,m,g,['concat','fv','state_official'],mask=mask);del mask;gc.collect()
    elif branch in ('context','context_state'):
        y,m,g=load_primary()
        for ctx in ('Rest','Stim8hr','Stim48hr'):
            for permission in ('P0','P1'):
                mm=m.copy();mm['partition']='excluded';other=mm.culture_condition.ne(ctx)
                mm.loc[other&mm.global_target_partition.eq('train'),'partition']='train';mm.loc[other&mm.global_target_partition.eq('validation'),'partition']='validation';mm.loc[~other&mm.global_target_partition.eq('test'),'partition']='test'
                assert not set(mm.loc[mm.partition.eq('train'),'target_contrast'])&set(mm.loc[mm.partition.isin(['validation','test']),'target_contrast'])
                assert not mm.loc[mm.partition.isin(['train','validation']),'culture_condition'].eq(ctx).any()
                families=['state_official'] if branch=='context_state' else ['zero','training_mean','ridge','concat','fv']
                run('whole_context_'+ctx+'_'+permission,y,mm,g,families,permission,stimulus=True)
    write_json(ROOT/(branch+'_status.json'),{'state':'complete','updated_epoch':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('branch',choices=['hierarchy','missing','context','context_state']);a=p.parse_args();lock=ROOT/(a.branch+'.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main(a.branch)
    finally:lock.unlink(missing_ok=True)
