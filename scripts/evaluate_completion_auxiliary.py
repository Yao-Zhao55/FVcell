"""Auxiliary metrics for the actual final checkpoints; no model updates."""
from completion_core import *
from full_suite_run import extra_primary_metrics,load_primary
def main():
    (ROOT/'analysis').mkdir(exist_ok=True)
    y,original,g=load_primary();out=ROOT/'auxiliary';out.mkdir(exist_ok=True);records=[]
    for task,permission in [('hierarchical_combo_P2','P2'),('hierarchical_global_P1','P1')]:
        m=pd.read_csv(ROOT/'runs'/task/'rows.csv');f=features(y,m,g,permission);ii=np.flatnonzero(m.partition.eq('test'));dc=np.zeros(len(m),np.int64)
        for family in ('hierarchical_pooled','fv_continued'):
            for seed in PLAN['seeds']:
                dest=out/task/family/str(seed);dest.mkdir(parents=True,exist_ok=True);p=dest/'metrics.json'
                if p.exists():value=json.loads(p.read_text())
                else:
                    cp=torch.load(ROOT/'runs'/task/family/str(seed)/'model.pt',weights_only=False,map_location='cpu');model=model_for(family,y.shape[1],1).cuda();model.load_state_dict(cp['state_dict'])
                    pred=prediction(model,f,ii,dc);value=extra_primary_metrics(m,ii,pred,dest);write_json(p,value);del cp,model,pred;gc.collect();torch.cuda.empty_cache()
                result=json.loads((ROOT/'runs'/task/family/str(seed)/'result.json').read_text())
                records.append({'task':task,'family':family,'seed':seed,'signed_Pearson':result['test']['pearson'],'direction_accuracy':result['test']['direction_accuracy'],'macro_context_DEG_AP':value['macro_context_AP'],'DEG_positive_prevalence':value['DEG_positive_prevalence'],'AP_undefined_rows_no_positives':value['AP_undefined_rows_no_positives']})
        del f;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(ROOT/'analysis/auxiliary_metrics.csv',index=False,encoding='utf-8-sig');write_json(ROOT/'auxiliary_status.json',{'state':'complete','updated_epoch':time.time()})
if __name__=='__main__':
    lock=ROOT/'auxiliary.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
