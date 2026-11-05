"""Audit all newly registered computations before interpreting their results."""
from completion_core import *
import datetime,importlib.metadata as md

def main():
    registry=json.loads((ROOT/'task_registry.json').read_text());rows=[];models=[];missing=[];manifests=[]
    for task,families in registry['tasks'].items():
        r=ROOT/'runs'/task
        if not (r/'rows.csv').exists():missing.append(task+'/rows.csv');continue
        m=pd.read_csv(r/'rows.csv');tr=m.partition.eq('train');va=m.partition.eq('validation');te=m.partition.eq('test')
        assert m['index'].is_unique and tr.any() and va.any() and te.any()
        assert m.loc[va].groupby('target_contrast').selection_or_calibration.nunique().eq(1).all()
        if task.startswith('hierarchical_donor'):
            assert m.groupby('donor_id').partition.nunique().eq(1).all()
            assert m.loc[tr,'donor_id'].nunique()==2 and m.loc[va,'donor_id'].nunique()==1 and m.loc[te,'donor_id'].nunique()==1
        elif task!='hierarchical_combo_P2':
            assert not set(m.loc[tr,'target_contrast'])&set(m.loc[va|te,'target_contrast'])
        if task.startswith('whole_context'):
            ctx=m.loc[te,'culture_condition'].unique();assert len(ctx)==1 and not m.loc[tr|va,'culture_condition'].isin(ctx).any()
        manifests.append({'task':task,'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'test_rows':int(te.sum()),'test_targets':int(m.loc[te,'target_contrast'].nunique()),'rows_sha256':sha(r/'rows.csv')})
        for family in families:
            dest=r/family;p=dest/'summary.json'
            if not p.exists():missing.append(task+'/'+family);continue
            s=json.loads(p.read_text())
            if s.get('deterministic'):
                f=pd.read_csv(dest/'test_rows.csv');assert set(f['index'])==set(m.loc[te,'index']) and f['index'].is_unique
                score=float(f.groupby(['donor','context']).mae.mean().groupby('donor').mean().mean());assert abs(score-s['test']['mae'])<1e-10
                rows.append({'task':task,'family':family,'mae':score,'seed_sd':0,'n_seeds':1,'parameters':None,'selected_fit_seconds':None,'test_rows':int(te.sum())});continue
            assert s['seeds']==[11,29,47]
            for seed,result in zip(s['seeds'],s['runs']):
                assert result['fit']['seed']==seed and not result['test_used_for_selection'];f=pd.read_csv(dest/str(seed)/'test_rows.csv');assert set(f['index'])==set(m.loc[te,'index']) and f['index'].is_unique
                score=float(f.groupby(['donor','context']).mae.mean().groupby('donor').mean().mean());assert abs(score-result['test']['mae'])<1e-10
                best=float('inf')
                for event in result['fit']['history']:
                    if event['validation_mae']<best-PLAN['min_delta']:best=event['validation_mae']
                assert abs(best-result['fit']['best_selection_mae'])<1e-12
                chosen=min(result['tuning_trials'],key=lambda t:t['best_selection_mae'])
                assert chosen['lr']==result['fit']['lr'] and chosen['strength']==result['fit']['strength']
                model=dest/str(seed)/'model.pt';cp=torch.load(model,map_location='cpu',weights_only=False);assert cp['protocol_sha256']==sha(ROOT/'protocol.json')
                if family in ('hierarchical','fixed_gaussian','unshrunk_offsets','hierarchical_pooled'):
                    assert bool((cp['state_dict']['state.weight'][:,32:]==0).all()),'Unregularized categorical bypass remains'
                    assert result['fit']['categorical_path']=='hierarchical_random_effect_only'
                    assert result['fit']['training_only_reparameterization_max_error']<3e-5
                if family=='fv_continued' or family in ('hierarchical','fixed_gaussian','unshrunk_offsets','hierarchical_pooled'):
                    assert cp['correction_protocol_sha256']==sha(ROOT/'identifiability_correction_protocol.json') and result['fit']['steps']<=1000
                if task.startswith('hierarchical_donor'):
                    assert set(cp['train_donor_mapping'])==set(m.loc[tr,'donor_id']);assert not set(cp['train_donor_mapping'])&set(m.loc[va|te,'donor_id'])
                models.append({'task':task,'family':family,'seed':seed,'mae':score,'lr':result['fit']['lr'],'strength':result['fit']['strength'],'model_path':str(model.relative_to(ROOT)),'sha256':sha(model)});del cp
            rows.append({'task':task,'family':family,'mae':s['mean_test_MAE'],'seed_sd':s['sd_test_MAE'],'n_seeds':3,'parameters':s['runs'][0]['fit']['parameters'],'selected_fit_seconds':sum(t['fit']['seconds'] for t in s['runs']),'test_rows':int(te.sum())})
    if missing:
        write_json(ROOT/'audit.json',{'state':'incomplete','missing':missing,'completed_checkpoints':len(models),'updated_epoch':time.time()});print(json.dumps({'missing':missing},indent=2));return
    if not (ROOT/'calibration_status.json').exists():
        write_json(ROOT/'audit.json',{'state':'models_complete_pending_calibration','missing':['reserved-donor calibration'],'completed_checkpoints':len(models)});print('Training models complete; donor interval evaluation remains');return
    if not (ROOT/'auxiliary_status.json').exists():
        write_json(ROOT/'audit.json',{'state':'models_complete_pending_auxiliary','missing':['final model auxiliary metrics'],'completed_checkpoints':len(models)});return
    n70=np.load(ROOT/'masks/observed_70.npy');n30=np.load(ROOT/'masks/observed_30.npy');assert not ((n30&np.bitwise_not(n70))!=0).any()
    protected=json.loads((BASE/'protected_files_before.json').read_text(encoding='utf8'));root=BASE.parents[2]
    assert all(sha(root/p['path'])==p['sha256'] for p in protected)
    frozen=json.loads((BASE/'prepared/FREEZE_LOCK.json').read_text());assert all(sha(BASE/'prepared'/n)==h for n,h in frozen['sha256'].items())
    an=ROOT/'analysis';an.mkdir(exist_ok=True);frame=pd.DataFrame(rows);frame.to_csv(an/'all_results.csv',index=False,encoding='utf-8-sig');pd.DataFrame(models).to_csv(an/'model_manifest.csv',index=False,encoding='utf-8-sig')
    # Four actual held-out donors; seed averages are inside each donor.
    donor=frame[frame.task.str.startswith('hierarchical_donor')].copy();donor['fold']=donor.task.str[-1].astype(int)
    previous=pd.read_csv(OUT/'analysis/donor_crossvalidation.csv')
    for family in ('fv','concat','state_official'):
        for row in previous[previous.family.eq(family)].itertuples():
            donor=pd.concat([donor,pd.DataFrame([{'task':'previous','family':'previous_'+family,'fold':row.fold,'mae':row.MAE}])],ignore_index=True)
    donor.to_csv(an/'donor_results.csv',index=False,encoding='utf-8-sig');pivot=donor.pivot(index='fold',columns='family',values='mae');contrasts={}
    for ref in ('fixed_gaussian','unshrunk_offsets','fv_continued','previous_fv','previous_concat','previous_state_official'):
        d=(pivot[ref]-pivot.hierarchical).to_numpy();sign=np.array(list(itertools.product([-1,1],repeat=4)));v=(sign*d).mean(1);p=float((np.abs(v)>=abs(d.mean())-1e-12).mean());draw=np.array(list(itertools.product(range(4),repeat=4)));b=d[draw].mean(1)
        contrasts[ref]={'absolute_MAE_reduction':float(d.mean()),'four_donor_reductions':d.tolist(),'exact_sign_flip_p':p,'descriptive_donor_bootstrap_95CI':np.quantile(b,[.025,.975]).tolist(),'statistical_warning':'n=4; exact numerical enumeration of 16 sign flips assumes sign symmetry. Overlapping CV training can violate independence; p and bootstrap intervals are descriptive, not guaranteed inferential tests or clinical/population precision.'}
    ordered=sorted(contrasts,key=lambda k:contrasts[k]['exact_sign_flip_p']);last=0
    for i,k in enumerate(ordered):last=max(last,min(1,(len(ordered)-i)*contrasts[k]['exact_sign_flip_p']));contrasts[k]['exploratory_Holm_p']=last
    write_json(an/'hierarchical_donor_contrasts.json',contrasts)
    old=pd.read_csv(OUT/'analysis/all_model_summary.csv');missing_rows=[]
    for family in ('concat','fv','state_official','hierarchical_pooled','fv_continued'):
        if family in ('hierarchical_pooled','fv_continued'):
            a=frame[frame.task.eq('hierarchical_global_P1')&frame.family.eq(family)].iloc[0];score=a.mae;sd=a.seed_sd
        else:
            a=old[old.task.eq('E1_global_target_P1')&old.family.eq(family)].iloc[0];score=a.mae;sd=a.seed_std
        missing_rows.append({'observed_fraction':1.,'family':family,'mae':score,'seed_sd':sd,'role':'full-label reference; matching original global-target split'})
        for n in (70,30):
            a=frame[frame.task.eq('missing_'+str(n)+'_global_P1')&frame.family.eq(family)].iloc[0];missing_rows.append({'observed_fraction':n/100,'family':family,'mae':a.mae,'seed_sd':a.seed_sd,'role':'nested artificial training-label deletion; actual test responses'})
    pd.DataFrame(missing_rows).to_csv(an/'missingness_results.csv',index=False,encoding='utf-8-sig')
    interval=[]
    for fold in range(4):
        for family in ('fixed_gaussian','hierarchical','unshrunk_offsets','fv_continued'):
            value=json.loads((ROOT/'calibration'/(str(fold)+'_'+family+'.json')).read_text())
            for context,result in value['by_context'].items():interval.append({'fold':fold,'family':family,'context':context,**result})
    pd.DataFrame(interval).to_csv(an/'donor_interval_results.csv',index=False,encoding='utf-8-sig')
    functional_joint(an)
    auxiliary=pd.read_csv(an/'auxiliary_metrics.csv');assert len(auxiliary)==12 and np.isfinite(auxiliary.macro_context_DEG_AP).all()
    files=[BASE/'scripts'/n for n in ['completion_core.py','run_training_completion.py','test_training_completion.py','queue_training_completion.py','analyse_training_completion.py','calibrate_training_completion.py','finish_corrective_evaluation.py','report_training_completion.py','run_pooled_completion.py','correct_context_ID_training.py','completion_metrics.py','evaluate_completion_auxiliary.py','qa_training_completion.py']]
    write_json(ROOT/'code_manifest.json',{'files':{p.name:sha(p) for p in files},'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'precision':'FP32','timing':'selected fits only; excludes discarded candidates and data/feature construction'})
    write_json(ROOT/'audit.json',{
        'state':'all_registered_additional_runs_complete','new_tasks':len(registry['tasks']),
        'new_method_task_settings':sum(map(len,registry['tasks'].values())),'new_neural_checkpoints':len(models),
        'new_deterministic_baseline_settings':sum(map(len,registry['tasks'].values()))-len(models)//3,
        'new_pipeline_checks_passed':14,'of_which_reused_checks':6,
        'protected_original_files_unchanged':len(protected),'old_frozen_split_hashes_unchanged':True,
        'nested_training_masks_checked':True,'categorical_bypass_removed_and_corrective_training_verified':True,
        'final_checkpoint_auxiliary_metric_runs':12,'held_donor_interval_cases':16,
        'registry_sha256':sha(ROOT/'task_registry.json'),'protocol_sha256':sha(ROOT/'protocol.json'),'manifests':manifests,
        'remaining':'fully matched broad pretraining, independent primary-T study and real organ dynamics require appropriate data; additions remain exploratory',
        'time':datetime.datetime.now().astimezone().isoformat()})
    print(json.dumps({'state':'complete','new_checkpoints':len(models),'new_tasks':len(registry['tasks'])}))
def functional_joint_family(an,family):
    from scipy.stats import spearmanr
    task=ROOT/'runs/hierarchical_functional_P1';m=pd.read_csv(task/'rows.csv');test=m[m.partition.eq('test')].reset_index(drop=True)
    pred=np.mean([np.load(task/family/str(s)/'test_predictions.npy') for s in PLAN['seeds']],axis=0)
    genes=pd.read_csv(PREP/'primary_genes.csv');source=pd.read_csv(OUT/'analysis/functional_matched_measurements.csv');source=source.groupby(['symbol','Donor'])[['IL10_delta_percentage_points','IL21_delta_percentage_points']].mean().reset_index();rows=[];points=[]
    for context in ('Rest','Stim8hr','Stim48hr'):
        ii=test.culture_condition.eq(context).to_numpy()
        for endpoint in ('IL10','IL21'):
            positions=np.flatnonzero(genes.gene_name.eq(endpoint));assert len(positions)==1
            rna=pd.DataFrame({'symbol':test.loc[ii,'target_contrast_gene_name'],'value':pred[ii,positions[0]]}).groupby('symbol').value.mean();protein=source.groupby('symbol')[endpoint+'_delta_percentage_points'].mean();shared=sorted(set(rna.index)&set(protein.index));x=rna.loc[shared].to_numpy();z=protein.loc[shared].to_numpy();rho=float(spearmanr(x,z).statistic) if np.std(x)>0 and np.std(z)>0 else None
            donors=source.Donor.unique();matrix=source.pivot(index='symbol',columns='Donor',values=endpoint+'_delta_percentage_points').reindex(index=shared,columns=donors).to_numpy();rng=np.random.default_rng(20261026);weights=np.zeros((2000,len(donors)))
            draws=rng.integers(0,len(donors),size=(2000,len(donors)))
            for j in range(2000):np.add.at(weights[j],draws[j],1)
            den=np.isfinite(matrix).astype(float)@weights.T;values=np.divide(np.nan_to_num(matrix)@weights.T,den,out=np.full(den.shape,np.nan),where=den>0);boot=[]
            for j in range(2000):
                picked=rng.integers(0,len(shared),len(shared));a=x[picked];b=values[picked,j];ok=np.isfinite(a)&np.isfinite(b)
                if ok.sum()>=3 and np.std(a[ok])>0 and np.std(b[ok])>0:boot.append(float(spearmanr(a[ok],b[ok]).statistic))
            ci=np.quantile(boot,[.025,.975]).tolist() if boot else [None,None]
            rows.append({'family':family,'RNA_context':context,'endpoint':endpoint,'targets':len(shared),'spearman':rho,'CI_low':ci[0],'CI_high':ci[1],'direction':float((np.sign(x)==np.sign(z)).mean()),'scope':'same-study preselected panel; extra assay/donors, not independent study; timings not harmonized'})
            points.extend({'family':family,'context':context,'endpoint':endpoint,'target':t,'RNA_log2FC':float(a),'protein_delta_percentage_points':float(b)} for t,a,b in zip(shared,x,z))
    return rows,points

def functional_joint(an):
    values=[functional_joint_family(an,f) for f in ('hierarchical_pooled','fv_continued')]
    pd.DataFrame([r for rows,_ in values for r in rows]).to_csv(an/'joint_functional_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([r for _,points in values for r in points]).to_csv(an/'joint_functional_points.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
