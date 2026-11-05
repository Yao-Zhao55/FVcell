"""Execute a frozen expanded model suite; resume only completed audited runs."""
from pathlib import Path
import os,sys,time,json,copy,traceback,gc
from full_suite_core import *
RUNS=OUT/'runs';RUNS.mkdir(exist_ok=True)

def freeze_plan():
    protocol={'version':'FVcell-outline-v2-expanded-20261026-v1','seeds':[11,29,47],'learning_rates':[.001,.0003],'max_steps':3000,'batch_size':128,'validation_interval':250,'patience':4,'loss':'SmoothL1_beta_0.1','residual_penalty':.001,'mechanism_penalty':.001,'main_metric':'signed response MAE; equal donor then context then target','primary_comparison':'factorized FV vs capacity-matched concatenation at P2','calibration':'half of original validation targets by pre-fixed SHA256; never early stopping or hyperparameter selection','prior_test_exposure':'first additive-run combination test summary was seen before this suite; expanded combination results are exploratory, not a newly untouched confirmation','separate_unseen_donor_unit':'delta_log2_CPM_plus_1 from raw-QC donor/guide counts, all reference genes, separate from pooled DESeq2 labels','tasks':{'E1_P2_combo':['fv','concat','shared','independent','state_official','ridge','zero','training_mean','same_target_mean'],'E2_P1_combo':['fv','concat','state_official','ridge'],'E2_P0_combo':['fv','concat','ridge'],'E1_global_target_P1':['fv','concat','state_official','ridge'],'E1_donor_P1':['fv','concat','shared','ridge'],'E2_donor_P2':['fv','concat','ridge'],'E4_ablations':['no_mechanism','randomized_mechanism','no_mechanism_regularizer','no_shrink','fixed_state','onehot_state'],'E3_internal_scale':['50_percent_aux_training','75_percent_aux_training'],'E5_functional_targets_P1':['fv','concat','ridge']},'E3_wide_data_scope':'K562/RPE1 non-primary-T backgrounds if published raw-count semantics validate; not a claim of a fully matched broad pretraining corpus','E5_readouts':['IL10_perc','IL21_perc'],'E5_all_three_RNA_contexts_reported_no_best_context_selection':True,'E5_functional_target_RNA_effects_removed_from_all_training_and_selection':'all contexts, no same-target teacher','no_pretrained_weights_or_old_NPZ':True,'download_budget_bytes':100_000_000_000,'no_organ_or_dynamics_claim_without_matched_data':True}
    p=OUT/'experiment_plan_frozen.json'
    if p.exists():
        if json.loads(p.read_text(encoding='utf-8'))!=protocol:raise ValueError('Frozen full-suite protocol differs; explicit amendment required')
    else:write_json(p,protocol)
    return protocol

def load_primary():
    y=np.load(BASE/'training_first/effects.npy',mmap_mode='r');meta=pd.read_csv(PREP/'primary_rows.csv');genes=pd.read_csv(PREP/'primary_genes.csv').gene_id.astype(str).to_numpy()
    meta['partition']=meta.combo_partition
    return y,meta,genes

def calibration(y,meta,cal,cp,test,tp,scale):
    summary={};scale=np.maximum(scale,.05)
    for ctx in sorted(meta.culture_condition.unique()):
        cv=meta.iloc[cal].culture_condition.to_numpy()==ctx;tv=meta.iloc[test].culture_condition.to_numpy()==ctx
        if not cv.any() or not tv.any():continue
        s=[]
        for j in np.flatnonzero(cv):s.append(float(np.max(np.abs(measured(y,meta,np.array([cal[j]]))[0]-cp[j])/scale)))
        n=len(s);rank=int(np.ceil((n+1)*.9));q=float(np.sort(s)[min(rank,n)-1]);covered=[];point=0;total=0
        for j in np.flatnonzero(tv):
            err=np.abs(measured(y,meta,np.array([test[j]]))[0]-tp[j]);ok=err<=scale*q;covered.append(bool(ok.all()));point+=int(ok.sum());total+=len(ok)
        summary[str(ctx)]={'calibration_rows':n,'calibration_order_stat_rank':rank,'simultaneous_response_vector_coverage':float(np.mean(covered)),'coordinate_coverage':point/total,'mean_interval_width':float(np.mean(2*scale*q)),'nominal':.9}
    return {'by_context':summary,'guarantee':'empirical coverage only; cross-condition/target dependence and donor shift do not establish exchangeability','calibration_targets_not_used_for_early_stopping_or_model_selection':True}

def extra_primary_metrics(meta,test,pred,run_dir):
    from sklearn.metrics import average_precision_score
    source=BASE/'large_data/GWCD4i.DE_stats.h5ad'
    from run_first_training import h5py,index
    with h5py.File(source,'r') as f:
        ids=index(f['obs']);loc={x:i for i,x in enumerate(ids)};record=[]
        for j,i in enumerate(test):
            p=f['layers/adj_p_value'][loc[str(meta.iloc[i]['index'])]];mask=np.isfinite(p);truth=p[mask]<=.1
            ap=float(average_precision_score(truth,np.abs(pred[j,mask]))) if truth.any() else None
            record.append({'index':str(meta.iloc[i]['index']),'context':str(meta.iloc[i].culture_condition),'average_precision':ap,'positive_prevalence':float(truth.mean()) if len(truth) else None,'n_observed_padj':int(mask.sum()),'n_positive':int(truth.sum())})
    frame=pd.DataFrame(record);frame.to_csv(run_dir/'test_DEG_AP.csv',index=False,encoding='utf-8-sig')
    # Retrieval compares only fixed held-out candidates within the same context.
    retrieval={}
    for ctx in sorted(meta.culture_condition.unique()):
        jj=np.flatnonzero(meta.iloc[test].culture_condition.to_numpy()==ctx)
        a=torch.tensor(pred[jj],device='cuda');b=torch.tensor(measured(np.load(BASE/'training_first/effects.npy',mmap_mode='r'),meta,test[jj]),device='cuda')
        a=a-a.mean(dim=1,keepdim=True);b=b-b.mean(dim=1,keepdim=True)
        sim=torch.nn.functional.normalize(a,dim=-1)@torch.nn.functional.normalize(b,dim=-1).T
        diagonal=sim.diag();better=(sim>diagonal[:,None]+1e-7).sum(dim=1);ties=(torch.abs(sim-diagonal[:,None])<=1e-7).sum(dim=1)
        rank=better.float()+(ties.float()+1)/2;top1=torch.where(better==0,1./ties.clamp_min(1),0.)
        retrieval[str(ctx)]={'candidates':len(jj),'average_tie_rank':float(rank.mean()),'top1_expected_with_ties':float(top1.mean())}
    return {'macro_context_AP':None if frame.average_precision.notna().sum()==0 else float(frame.groupby('context').average_precision.mean().mean()),'DEG_positive_prevalence':float(frame.positive_prevalence.mean()),'AP_undefined_rows_no_positives':int(frame.average_precision.isna().sum()),'ranking_score':'absolute predicted log2FC; all finite measured adjusted p-values; FDR <= 0.1','within_context_perturbation_retrieval':retrieval}

def baseline_predictions(y,meta,f,test,kind):
    if kind=='zero':return np.zeros((len(test),y.shape[1]),np.float32)
    if kind=='training_mean':return np.tile(f['mean'],(len(test),1))
    if kind=='same_target_mean':
        out=np.empty((len(test),y.shape[1]),np.float32)
        for j,i in enumerate(test):
            t=f['target_codes'][i];out[j]=f['target_sums'][t]/f['target_counts'][t] if f['target_counts'][t] else f['mean']
        return out
    raise ValueError(kind)

def ridge(y,meta,f,protocol,folder):
    train=np.flatnonzero(f['train_mask']);select=np.flatnonzero((meta.partition=='validation')&(meta.selection_or_calibration=='selection'));test=np.flatnonzero(meta.partition=='test')
    x=np.concatenate([f['u'],f['z'],np.ones((len(meta),1),np.float32)],axis=1)
    xtx=np.zeros((x.shape[1],x.shape[1]),np.float64);xty=np.zeros((x.shape[1],y.shape[1]),np.float64)
    for start in range(0,len(train),512):
        ii=train[start:start+512];a=x[ii].astype(np.float64);b=measured(y,meta,ii).astype(np.float64)-f['mean'];xtx+=a.T@a;xty+=a.T@b
    choices=[];best=None
    for alpha in (.1,10.,1000.):
        reg=np.eye(x.shape[1])*alpha;reg[-1,-1]=0;w=np.linalg.solve(xtx+reg+np.eye(x.shape[1])*1e-8,xty).astype(np.float32)
        p=x[select]@w+f['mean'];v=scores(y,meta,select,p)['mae'];choices.append({'alpha':alpha,'selection_mae':v})
        if best is None or v<best[0]:best=(v,alpha,w)
    pred=x[test]@best[2]+f['mean'];result=scores(y,meta,test,pred,folder/'test_rows.csv')
    np.savez_compressed(folder/'ridge_model.npz',weights=best[2],training_mean=f['mean'],**f['transform'])
    write_json(folder/'result.json',{'model':'ridge','choices':choices,'selected_alpha':best[1],'test':result,'parameter_count':int(best[2].size),'test_used_for_selection':False})
    return result

def run_task(name,y,meta,genes,families,protocol,permission='P2'):
    task=RUNS/name;task.mkdir(exist_ok=True);meta.to_csv(task/'rows.csv',index=False,encoding='utf-8-sig')
    test=np.flatnonzero(meta.partition=='test');cal=np.flatnonzero((meta.partition=='validation')&(meta.selection_or_calibration=='calibration'))
    if not len(test) or not len(cal):raise ValueError('Missing fixed test/calibration rows')
    real=features(y,meta,genes,permission);feature_cache={'real':real}
    write_json(task/'information_audit.json',{'permission':permission,'target_effect_rows_used_for_features':'only partition=train; train features leave own response out' if permission=='P2' else 'none','gene_unit':'published_DESeq2_log2FC' if 'storage_row' not in meta else 'delta_log2_CPM_plus_1','train_rows':int(real['train_mask'].sum()),'test_rows':len(test),'selection_validation_rows':int(((meta.partition=='validation')&(meta.selection_or_calibration=='selection')).sum()),'calibration_rows':len(cal),'training_ID_sha256':__import__('hashlib').sha256('\n'.join(meta.loc[real['train_mask'],'index'].astype(str)).encode()).hexdigest(),'frozen_plan_sha256':sha(OUT/'experiment_plan_frozen.json'),'source_row_manifest_sha256':sha(task/'rows.csv')})
    for family in families:
        folder=task/family;folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():continue
        write_json(OUT/os.environ.get('FVCELL_STATUS_NAME','suite_status.json'),{'state':'running','task':name,'family':family,'updated_epoch':time.time()})
        if family in ('zero','training_mean','same_target_mean'):
            pred=baseline_predictions(y,meta,real,test,family);summary=scores(y,meta,test,pred,folder/'test_rows.csv');write_json(folder/'summary.json',{'deterministic':True,'test':summary});continue
        if family=='ridge':
            result=ridge(y,meta,real,protocol,folder);write_json(folder/'summary.json',{'deterministic':True,'test':result});continue
        mk='id_random' if family=='no_mechanism' else 'randomized' if family=='randomized_mechanism' else 'real'
        if mk not in feature_cache:feature_cache[mk]=features(y,meta,genes,permission,mk)
        f=feature_cache[mk]
        if family=='onehot_state':f=dict(f);f['z']=f['z'].copy();f['z'][:,:32]=0
        selected_lr=None;all_results=[]
        for seed in protocol['seeds']:
            seed_dir=folder/str(seed);seed_dir.mkdir(exist_ok=True)
            existing=seed_dir/'result.json'
            if existing.exists():
                old=json.loads(existing.read_text(encoding='utf-8'));all_results.append(old);selected_lr=old['fit']['lr'];continue
            lrs=protocol['learning_rates'] if selected_lr is None else [selected_lr]
            best=None
            for lr in lrs:
                model,info=fit(y,meta,f,family,seed,lr,seed_dir,protocol)
                if best is None or info['best_selection_mae']<best[1]['best_selection_mae']:
                    best=(copy.deepcopy(model).cpu(),info)
                del model;gc.collect();torch.cuda.empty_cache()
            model,info=best;model=model.cuda();selected_lr=info['lr']
            pred=predict(model,f,test,f['mean']);summary=scores(y,meta,test,pred,seed_dir/'test_rows.csv')
            cp=predict(model,f,cal,f['mean']);ci=calibration(y,meta,cal,cp,test,pred,f['std'])
            torch.save({'state_dict':model.cpu().state_dict(),'family':family,'gene_ids':genes.tolist(),'permission':permission,'mean':f['mean'],'feature_transform':f['transform'],'seed':seed,'fit':info,'plan_sha256':sha(OUT/'experiment_plan_frozen.json')},seed_dir/'model.pt')
            value={'fit':info,'test':summary,'calibration':ci,'test_used_for_model_selection':False}
            if 'storage_row' not in meta:
                value['auxiliary_metrics']=extra_primary_metrics(meta,test,pred,seed_dir)
                if name=='E1_P2_combo' and family in ('fv','concat'):
                    np.save(seed_dir/'test_predictions.npy',pred)
                    vpidx=np.flatnonzero(meta.partition=='validation');vp=predict(model.cuda(),f,vpidx,f['mean']);np.save(seed_dir/'validation_predictions.npy',vp)
                elif name.startswith('E5_'):np.save(seed_dir/'test_predictions.npy',pred)
            elif name=='E1_external_study_P1' and family in ('fv','concat'):
                np.save(seed_dir/'test_predictions.npy',pred)
                vi=np.flatnonzero(meta.partition=='validation');np.save(seed_dir/'validation_predictions.npy',predict(model.cuda(),f,vi,f['mean']))
            write_json(existing,value);all_results.append(value)
            del model,pred,cp;gc.collect();torch.cuda.empty_cache()
        values=[v['test']['mae'] for v in all_results]
        write_json(folder/'summary.json',{'test_mae_seed_mean':float(np.mean(values)),'test_mae_seed_std':float(np.std(values,ddof=1)) if len(values)>1 else 0.,'seeds':protocol['seeds'],'parameter_count':all_results[0]['fit']['parameter_count'],'selected_lr_first_seed':all_results[0]['fit']['lr'],'seed_variability_is_not_biological_replication':True,'runs':all_results})
    write_json(task/'complete.json',{'state':'complete','families':families,'updated_epoch':time.time()})
    del feature_cache,real;gc.collect();torch.cuda.empty_cache()

def main():
    lock=OUT/'suite.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        protocol=freeze_plan();y,m,g=load_primary()
        families=['zero','training_mean','same_target_mean','ridge','shared','concat','independent','state_official','fv','no_mechanism','randomized_mechanism','no_mechanism_regularizer','no_shrink','fixed_state','onehot_state']
        run_task('E1_P2_combo',y,m.copy(),g,families,protocol,'P2')
        for p in ('P1','P0'):run_task('E2_'+p+'_combo',y,m.copy(),g,['ridge','concat','fv']+(['state_official'] if p=='P1' else []),protocol,p)
        global_meta=m.copy();global_meta['partition']=global_meta.global_target_partition
        run_task('E1_global_target_P1',y,global_meta,g,['ridge','concat','state_official','fv'],protocol,'P1')
        # Freeze supportive other-context evidence for all held targets when scaling.
        held=set(m.loc[m.partition!='train','target_contrast']);train=m.partition.eq('train');required=train&m.target_contrast.isin(held)
        for fraction in (.5,.75):
            mm=m.copy();budget=int(train.sum()*fraction);candidates=np.flatnonzero(train&~required);order=sorted(candidates,key=lambda i:hashnum('scale|'+str(mm.iloc[i].target_contrast)))
            keep=np.zeros(len(m),bool);keep[np.flatnonzero(required)]=True;keep[order[:max(0,budget-int(required.sum()))]]=True;mm.loc[train&~keep,'partition']='excluded'
            run_task('E3_internal_'+str(int(fraction*100)),y,mm,g,['fv'],protocol,'P2')
        # Independent measured readouts: withhold every RNA effect of these targets.
        flow=pd.read_csv(BASE/'sources/GWT_IL10_IL21_arrayed_validation_combined.csv');symbols=set(flow.Perturbation.str.replace(r'-\d+$','',regex=True))-{'NTC','NTC1','NTC2','NTC3'}
        functional=set(m.loc[m.target_contrast_gene_name.isin(symbols),'target_contrast']);mm=m.copy();mm.loc[mm.target_contrast.isin(functional),'partition']='test';mm.loc[(mm.partition=='test')&~mm.target_contrast.isin(functional),'partition']='excluded'
        write_json(OUT/'functional_scope_lock.json',{'target_symbols':sorted(symbols),'mapped_target_ids':sorted(functional),'RNA_test_conditions':sorted(mm.culture_condition.unique()),'protein_labels_used_for_training_or_selection':False,'RNA_effects_of_functional_targets_removed_from_all_training_and_selection':True,'assay_contexts_not_assumed_identical':True})
        run_task('E5_functional_targets_P1',y,mm,g,['ridge','concat','fv'],protocol,'P1')
        donor=np.load(PREP/'donor_effects.npy',mmap_mode='r');dm=pd.read_csv(PREP/'donor_rows.csv');dg=pd.read_csv(PREP/'donor_genes.csv').gene_id.astype(str).to_numpy()
        dm['selection_or_calibration']=['calibration' if hashnum('calibration|'+str(t))%2 else 'selection' for t in dm.target_contrast]
        for p,fs in [('P1',['ridge','shared','concat','fv']),('P2',['ridge','concat','fv'])]:run_task('E1_donor_'+p,donor,dm.copy(),dg,fs,protocol,p)
        write_json(OUT/'suite_status.json',{'state':'core_suite_complete','remaining':'E3 external source-matched branch and E4/E5 aggregate audits/plots','updated_epoch':time.time()})
    except BaseException as e:
        write_json(OUT/'suite_status.json',{'state':'blocked_or_failed','error':str(e),'traceback':traceback.format_exc(),'updated_epoch':time.time()});raise
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
