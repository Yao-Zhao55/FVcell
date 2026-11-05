"""Aggregate frozen tests, calibration, selective risk, and independent flow data."""
from full_suite_core import *
from full_suite_run import load_primary,calibration
from scipy.stats import spearmanr
import gc
ANALYSIS=OUT/'analysis';ANALYSIS.mkdir(exist_ok=True)

def collect():
    records=[]
    for task in sorted((OUT/'runs').iterdir()):
        if not task.is_dir():continue
        for family in sorted(task.iterdir()):
            p=family/'summary.json'
            if not p.exists():continue
            s=json.loads(p.read_text(encoding='utf-8'))
            if s.get('deterministic'):
                ridge_record=json.loads((family/'result.json').read_text()) if family.name=='ridge' and (family/'result.json').exists() else {}
                records.append({'task':task.name,'family':family.name,'mae':s['test']['mae'],'seed_std':0.,'n_seeds':1,'parameters':ridge_record.get('parameter_count',0 if family.name=='zero' else None),'rows':s['test']['rows'],'total_fit_seconds':None})
            else:
                created=getattr(family.stat(),'st_birthtime',family.stat().st_ctime);elapsed=max(0,p.stat().st_mtime-created)
                records.append({'task':task.name,'family':family.name,'mae':s['test_mae_seed_mean'],'seed_std':s['test_mae_seed_std'],'n_seeds':len(s['runs']),'parameters':s['parameter_count'],'rows':s['runs'][0]['test']['rows'],'total_fit_seconds':sum(r['fit']['elapsed_seconds'] for r in s['runs']),'family_wall_seconds_including_tuning_evaluation_and_recovery':elapsed})
    f=pd.DataFrame(records);f.to_csv(ANALYSIS/'all_model_summary.csv',index=False,encoding='utf-8-sig');return f

def paired_comparisons():
    out={}
    for task in sorted((OUT/'runs').iterdir()):
        if not (task/'fv/summary.json').exists():continue
        base=task/'concat'
        if not (base/'summary.json').exists():continue
        def load_family(name):
            pieces=[]
            for seed in (11,29,47):
                p=task/name/str(seed)/'test_rows.csv'
                if not p.exists():return None
                pieces.append(pd.read_csv(p).set_index('index'))
            ref=pieces[0];assert all(set(x.index)==set(ref.index) for x in pieces)
            mean=np.mean([p.loc[ref.index].mae.to_numpy() for p in pieces],axis=0)
            return pd.DataFrame({'mae':mean,'context':ref.context,'donor':ref.donor,'target':ref.target},index=ref.index)
        a=load_family('fv');b=load_family('concat')
        if a is None or b is None:continue
        b=b.loc[a.index];delta=b.mae-a.mae
        point=float(delta.groupby([a.donor,a.context]).mean().groupby('donor').mean().mean())
        # Resample target clusters jointly across contexts and fixed donors.
        targets=sorted(a.target.unique());target_context=delta.groupby([a.target,a.donor,a.context]).mean().unstack(['donor','context']).reindex(targets).to_numpy()
        rng=np.random.default_rng(20261026);draws=[]
        for _ in range(2000):
            sampled=target_context[rng.integers(0,len(targets),size=len(targets))]
            draws.append(float(np.nanmean(np.nanmean(sampled,axis=0))))
        out[task.name]={'contrast':'concat MAE minus FV MAE; positive favors FV','absolute_reduction':point,'target_cluster_bootstrap_95_CI':np.quantile(draws,[.025,.975]).tolist(),'resamples':2000,'fixed_seed_averaged_losses_not_ensemble_prediction_loss':True,'biological_replicate_warning':'conditional on the published cohort/fixed test donor; target clusters are not new independent human donors; multiple contrasts descriptive, not multiplicity-controlled significance claims'}
    write_json(ANALYSIS/'paired_model_comparisons.json',out)

def uncertainty():
    task=OUT/'runs/E1_P2_combo';meta=pd.read_csv(task/'rows.csv');y=np.load(BASE/'training_first/effects.npy',mmap_mode='r')
    vi=np.flatnonzero(meta.partition=='validation');ti=np.flatnonzero(meta.partition=='test');cal_mask=meta.iloc[vi].selection_or_calibration.eq('calibration').to_numpy();selection=~cal_mask
    summaries={};risk=[]
    for family in ('fv','concat'):
        paths=[task/family/str(s) for s in (11,29,47)]
        if not all((p/'test_predictions.npy').exists() and (p/'validation_predictions.npy').exists() for p in paths):continue
        tp=np.stack([np.load(p/'test_predictions.npy') for p in paths]);vp=np.stack([np.load(p/'validation_predictions.npy') for p in paths])
        test_pred=tp.mean(axis=0);val_pred=vp.mean(axis=0)
        f=features(y,meta,pd.read_csv(PREP/'primary_genes.csv').gene_id.to_numpy(),'P2')
        summaries[family]=calibration(y,meta,vi[cal_mask],val_pred[cal_mask],ti,test_pred,f['std'])
        vp_uncert=np.sqrt(vp.var(axis=0).mean(axis=1));tp_uncert=np.sqrt(tp.var(axis=0).mean(axis=1))
        center_val=np.linalg.norm(np.concatenate([f['u'][vi],f['z'][vi]],axis=1),axis=1);center_test=np.linalg.norm(np.concatenate([f['u'][ti],f['z'][ti]],axis=1),axis=1)
        truth=measured(y,meta,ti);err=np.abs(truth-test_pred).mean(axis=1)
        for method,sv,st in [('three_seed_disagreement',vp_uncert,tp_uncert),('training_feature_center_distance',center_val,center_test)]:
            for retain in (.2,.4,.6,.8,1.):
                threshold=float(np.quantile(sv[selection],retain));keep=st<=threshold
                values=pd.DataFrame({'error':err[keep],'context':meta.iloc[ti].culture_condition.to_numpy()[keep]})
                risk.append({'family':family,'support_score':method,'validation_retention_target':retain,'threshold_selected_without_test_labels':threshold,'actual_test_retention':float(keep.mean()),'retained_rows':int(keep.sum()),'retained_macro_context_MAE':None if not keep.any() else float(values.groupby('context').error.mean().mean())})
        del tp,vp,truth,f;gc.collect()
    pd.DataFrame(risk).to_csv(ANALYSIS/'selective_prediction.csv',index=False,encoding='utf-8-sig');write_json(ANALYSIS/'ensemble_calibration.json',summaries)

def functional():
    task=OUT/'runs/E5_functional_targets_strict_P1'
    if not (task/'complete.json').exists():return
    meta=pd.read_csv(task/'rows.csv');ti=np.flatnonzero(meta.partition=='test');testmeta=meta.iloc[ti].reset_index(drop=True)
    genes=pd.read_csv(PREP/'primary_genes.csv')
    for endpoint in ('IL10','IL21'):
        assert int(genes.gene_name.eq(endpoint).sum())==1,'Functional response symbol is not uniquely mapped: '+endpoint
    gene_index={str(v):i for i,v in enumerate(genes.gene_name)}
    flow=pd.read_csv(BASE/'sources/GWT_IL10_IL21_arrayed_validation_combined.csv');control=flow.Perturbation.str.match(r'^NTC\d*$')
    base=flow[control].groupby(['Donor','Batch'])[['IL10_perc','IL21_perc']].mean().rename(columns=lambda c:'control_'+c)
    assays=flow[~control].join(base,on=['Donor','Batch']);assays['symbol']=assays.Perturbation.str.replace(r'-\d+$','',regex=True)
    assays['missing_matched_control']=assays.control_IL10_perc.isna()|assays.control_IL21_perc.isna()
    assays=assays[~assays.missing_matched_control].copy()
    for e in ('IL10','IL21'):assays[e+'_delta_percentage_points']=assays[e+'_perc']-assays['control_'+e+'_perc']
    assays.to_csv(ANALYSIS/'functional_matched_measurements.csv',index=False,encoding='utf-8-sig')
    donor_meta=pd.read_csv(BASE/'sources/GWT_validation_donor_metadata.csv',dtype=str).set_index('donor_name');primary_donors=set(pd.read_csv(BASE/'sources/GWT_sample_metadata.suppl_table.csv').donor_id.astype(str))
    functional_ids={str(donor_meta.loc[d,'donor_id']) for d in flow.Donor.unique() if d in donor_meta.index}
    audit={'flow_rows':len(flow),'matched_perturbed_measurements':len(assays),'flow_donors':int(flow.Donor.nunique()),'functional_donor_ids_resolved':len(functional_ids),'donor_overlap_with_primary':sorted(functional_ids&primary_donors),'gene_targets':int(assays.symbol.nunique()),'role':'independent measured assay and resolved additional donors within same study; not independent study','assay_contexts_or_time_not_asserted_identical_to_RNA_contexts':True,'all_RNA_contexts_reported_not_selected_using_flow_labels':True,'preselected_validation_targets_are_not_an_unbiased_genome_wide_functional_test':True,'RNA_targets_removed_from_all_training_and_selection':json.loads((OUT/'functional_scope_lock.json').read_text())}
    source=assays.groupby(['symbol','Donor'])[['IL10_delta_percentage_points','IL21_delta_percentage_points']].mean().reset_index()
    records=[];points=[]
    for family in ('fv','concat'):
        files=[task/family/str(s)/'test_predictions.npy' for s in (11,29,47)]
        if not all(p.exists() for p in files):continue
        pred=np.mean([np.load(p) for p in files],axis=0)
        for context in sorted(testmeta.culture_condition.unique()):
            subset=testmeta.culture_condition.eq(context).to_numpy()
            for endpoint in ('IL10','IL21'):
                if endpoint not in gene_index or (genes.gene_name.astype(str)==endpoint).sum()!=1:
                    records.append({'family':family,'RNA_context':context,'endpoint':endpoint,'status':'response_gene_not_in_published_gene_universe'});continue
                prediction=pd.DataFrame({'symbol':testmeta.loc[subset,'target_contrast_gene_name'],'prediction':pred[subset,gene_index[endpoint]]}).groupby('symbol').prediction.mean()
                values=source.groupby('symbol')[endpoint+'_delta_percentage_points'].mean();values=values[np.isfinite(values)];prediction=prediction[np.isfinite(prediction)];shared=sorted(set(prediction.index)&set(values.index));x=prediction.loc[shared].to_numpy();z=values.loc[shared].to_numpy()
                if len(shared)<3:
                    records.append({'family':family,'RNA_context':context,'endpoint':endpoint,'status':'fewer_than_three_complete_target_pairs','targets':len(shared)});continue
                rho=float(spearmanr(x,z).statistic) if len(shared)>=3 and np.std(x)>0 and np.std(z)>0 else None
                rng=np.random.default_rng(20261026);boot=[];donors=source.Donor.unique()
                matrix=source.pivot(index='symbol',columns='Donor',values=endpoint+'_delta_percentage_points').reindex(index=shared,columns=donors).to_numpy()
                draws=rng.integers(0,len(donors),size=(2000,len(donors)));weights=np.zeros((2000,len(donors)),np.float64)
                for j in range(2000):np.add.at(weights[j],draws[j],1)
                den=np.isfinite(matrix).astype(float)@weights.T
                dvalues=np.divide(np.nan_to_num(matrix)@weights.T,den,out=np.full(den.shape,np.nan),where=den>0)
                for j in range(2000):
                    picked=rng.integers(0,len(shared),size=len(shared));xx=x[picked];zz=dvalues[picked,j];ok=np.isfinite(xx)&np.isfinite(zz)
                    if ok.sum()>=3 and np.std(xx[ok])>0 and np.std(zz[ok])>0:boot.append(float(spearmanr(xx[ok],zz[ok]).statistic))
                ci=np.quantile(boot,[.025,.975]).tolist() if boot else [None,None]
                records.append({'family':family,'RNA_context':context,'endpoint':endpoint,'targets':len(shared),'spearman_RNA_vs_protein_fraction_delta':rho,'two_way_target_and_functional_donor_bootstrap_95CI_low':ci[0],'CI_high':ci[1],'direction_agreement':float(np.mean(np.sign(x)==np.sign(z))) if len(shared) else None,'biological_unit_warning':'small preselected target panel, same study, context/timing mismatch; exploratory cross-assay concordance, not a calibrated protein prediction'})
                for symbol,xx,zz in zip(shared,x,z):points.append({'family':family,'RNA_context':context,'endpoint':endpoint,'target':symbol,'RNA_prediction_log2FC':float(xx),'protein_delta_percentage_points':float(zz)})
    pd.DataFrame(records).to_csv(ANALYSIS/'functional_concordance.csv',index=False,encoding='utf-8-sig');pd.DataFrame(points).to_csv(ANALYSIS/'functional_points.csv',index=False,encoding='utf-8-sig');write_json(ANALYSIS/'functional_evidence_audit.json',audit)

def donor_crossvalidation():
    import itertools
    records=[]
    for fold in range(4):
        name='E1_donor_P1' if fold==0 else 'E1_donor_P1_fold'+str(fold)
        task=OUT/'runs'/name
        for family in ('ridge','shared','concat','state_official','fv'):
            path=task/family/'summary.json'
            if not path.exists():continue
            s=json.loads(path.read_text())
            if s.get('deterministic'):
                score=s['test']['mae'];donor=next(iter(s['test']['by_donor_mae']))
            else:score=s['test_mae_seed_mean'];donor=next(iter(s['runs'][0]['test']['by_donor_mae']))
            records.append({'fold':fold,'test_donor':donor,'family':family,'MAE':score})
    frame=pd.DataFrame(records);frame.to_csv(ANALYSIS/'donor_crossvalidation.csv',index=False,encoding='utf-8-sig')
    contrasts={};pivot=frame.pivot(index='test_donor',columns='family',values='MAE')
    if len(pivot)==4 and 'fv' in pivot:
        for baseline in ('concat','shared','ridge','state_official'):
            if baseline not in pivot or pivot[baseline].isna().any() or pivot.fv.isna().any():continue
            d=(pivot[baseline]-pivot.fv).to_numpy();point=float(d.mean())
            signs=np.array(list(itertools.product([-1,1],repeat=4)))
            sign_values=(signs*d).mean(axis=1);p=float(np.mean(np.abs(sign_values)>=abs(point)-1e-12))
            draw=np.array(list(itertools.product(range(4),repeat=4)));boots=d[draw].mean(axis=1)
            contrasts[baseline]={'paired_held_donor_MAE_reduction':point,'four_donor_values':{str(k):float(v) for k,v in zip(pivot.index,d)},'exact_16_sign_flip_two_sided_p':p,'descriptive_256_donor_bootstrap_95CI':np.quantile(boots,[.025,.975]).tolist(),'caution':'Only four biological donors; overlapping crossvalidation training sets and upstream curated target selection remain; no large-population or independent-study guarantee'}
    write_json(ANALYSIS/'donor_level_comparisons.json',contrasts)

def decomposition_identity():
    rng=np.random.default_rng(20261026);p0=rng.dirichlet(np.ones(4),size=10000);p1=rng.dirichlet(np.ones(4),size=10000);u0=rng.lognormal(size=(10000,4));u1=rng.lognormal(size=(10000,4))
    total=(p1*u1).sum(axis=1)-(p0*u0).sum(axis=1);within=(p1*(u1-u0)).sum(axis=1);composition=((p1-p0)*u0).sum(axis=1)
    write_json(ANALYSIS/'composition_identity_demo.json',{'role':'analytical/synthetic consistency demonstration only, not an organ experiment or biological model accuracy result','draws':10000,'seed':20261026,'max_absolute_identity_error':float(np.max(np.abs(total-within-composition))),'nonidentifiability_example':{'baseline':{'p':[.5,.5],'mu':[0,2],'mixture':1},'scenario_A':{'p':[.5,.5],'mu':[1,3],'mixture':2,'within':1,'composition':0},'scenario_B':{'p':[0,1],'mu':[0,2],'mixture':2,'within':0,'composition':1}},'conclusion':'The same mixture change admits distinct within-type and composition explanations without independent observations; log-scale identity is not claimed'})

def main():
    collect();paired_comparisons();uncertainty();functional();donor_crossvalidation();decomposition_identity()
    write_json(ANALYSIS/'analysis_status.json',{'state':'complete_for_available_runs','updated_epoch':time.time()})
if __name__=='__main__':main()
