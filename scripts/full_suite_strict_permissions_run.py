from full_suite_core import *
from full_suite_run import load_primary,freeze_plan,run_task
def main():
    y,m,g=load_primary();protocol=freeze_plan()
    held=set(m.loc[m.partition.isin(['validation','test']),'target_contrast'])
    evidence=m.partition.eq('train').to_numpy();removed=evidence&m.target_contrast.isin(held).to_numpy()
    m.loc[removed,'partition']='evidence_only'
    for p in ('P0','P1','P2'):
        mm=m.copy();mm.attrs['evidence_mask']=evidence
        if p!='P2':
            assert not set(mm.loc[mm.partition=='train','target_contrast'])&held
        run_task('E2_strict_'+p,y,mm,g,['ridge','concat','fv'],protocol,p)
    write_json(OUT/'strict_permission_status.json',{'state':'complete','common_supervised_train_rows':int((m.partition=='train').sum()),'P2_additional_other_context_evidence_rows':int(removed.sum()),'strict_protocol_sha256':sha(OUT/'E2_strict_protocol_amendment.json'),'updated_epoch':time.time()})
if __name__=='__main__':main()
