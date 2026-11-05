"""Re-evaluate the existing additive rule under this suite's selection/calibration split.

The family and its 4x4 grid already existed before the expanded tests. This is
an exploratory benchmark audit, not a new method optimized from the test set.
"""
from full_suite_core import *
from full_suite_run import load_primary,calibration
from run_first_training import fit_shared,shared_row,mean_finite

def main():
    y,m,g=load_primary();task=OUT/'runs/E1_P2_combo'
    protocol={'role':'existing minimal additive comparator; rerun to match the expanded validation/calibration split','shared_weights':[0.,.25,.5,1.],'residual_weights':[0.,.25,.5,1.],'test_results_of_existing_family_were_previously_seen':True,'grid_unchanged_from_first_training':True,'no_test_based_selection':True}
    write_json(OUT/'existing_additive_baseline_protocol.json',protocol)
    tr=m.partition.eq('train').to_numpy();vi=np.flatnonzero(m.partition.eq('validation')&m.selection_or_calibration.eq('selection'));ci=np.flatnonzero(m.partition.eq('validation')&m.selection_or_calibration.eq('calibration'));ti=np.flatnonzero(m.partition.eq('test'))
    targets,codes=np.unique(m.target_contrast,return_inverse=True);contexts,ctxcodes=np.unique(m.culture_condition,return_inverse=True)
    sums,counts=fit_shared(y,codes,tr);mean=mean_finite(y[tr]);programs=np.zeros((3,len(g)),np.float64);n=np.zeros(3,int)
    for i in np.flatnonzero(tr):programs[ctxcodes[i]]+=y[i]-shared_row(y,i,codes,tr,sums,counts,mean,True);n[ctxcodes[i]]+=1
    programs=(programs/n[:,None]).astype(np.float32)
    def predict(indices,a,b):
        s=np.stack([shared_row(y,i,codes,tr,sums,counts,mean) for i in indices]);return (a*s+(1-a)*mean+b*programs[ctxcodes[indices]]).astype(np.float32)
    for family,weights in [('shrunk_additive',protocol['residual_weights']),('shrunk_shared',[0.])]:
        dest=task/family;dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():continue
        choices=[]
        for a in protocol['shared_weights']:
            for b in weights:choices.append({'shared_weight':a,'residual_weight':b,'selection_MAE':selection_mae(y,m,vi,predict(vi,a,b))})
        best=min(choices,key=lambda r:r['selection_MAE']);p=predict(ti,best['shared_weight'],best['residual_weight']);test=scores(y,m,ti,p,dest/'test_rows.csv')
        np.savez_compressed(dest/'model.npz',training_target_sums=sums,training_target_counts=counts,global_training_mean=mean,context_programs=programs,target_ids=targets.astype(str),context_names=contexts.astype(str),gene_ids=g.astype(str),shared_weight=best['shared_weight'],residual_weight=best['residual_weight'])
        v=predict(ci,best['shared_weight'],best['residual_weight']);scale=np.std(y[tr],axis=0)
        write_json(dest/'result.json',{'choices':choices,'selected':best,'test':test,'test_used_for_selection':False,'model':'existing shrunk shared effect plus optional rank<=3 context programs','calibration':calibration(y,m,ci,v,ti,p,scale),'protocol_sha256':sha(OUT/'existing_additive_baseline_protocol.json')})
        write_json(dest/'summary.json',{'deterministic':True,'test':test,'model_fitted_from_training_only':True,'hyperparameters_selected_on_selection_half_only':True})
    write_json(OUT/'existing_additive_baseline_status.json',{'state':'complete','updated_epoch':time.time()})
if __name__=='__main__':main()
