"""Independent manifest/result checks; no fitting or test-based model selection."""
from full_suite_core import *
from run_first_training import h5py,index,column,rows
import importlib.metadata as md,datetime,ctypes

def main():
    (OUT/'analysis').mkdir(exist_ok=True)
    expected={'E1_P2_combo':['zero','training_mean','same_target_mean','ridge','shared','concat','independent','state_official','fv','no_mechanism','randomized_mechanism','no_mechanism_regularizer','no_shrink','fixed_state','onehot_state'],'E2_P1_combo':['ridge','concat','state_official','fv'],'E2_P0_combo':['ridge','concat','fv'],'E1_global_target_P1':['ridge','concat','state_official','fv'],'E3_internal_50':['fv'],'E3_internal_75':['fv'],'E5_functional_targets_P1':['ridge','concat','fv'],'E1_donor_P1':['ridge','shared','concat','state_official','fv'],'E1_donor_P2':['ridge','concat','fv'],'E3_external_targeted':['ridge','concat','fv'],'E3_external_non_primary_T':['ridge','concat','fv'],'E1_external_study_P1':['zero','training_mean','ridge','concat','state_official','fv']}
    for i in (1,2,3):expected['E1_donor_P1_fold'+str(i)]=['ridge','shared','concat','state_official','fv']
    for p in ('P0','P1','P2'):expected['E2_strict_'+p]=['ridge','concat','fv']
    expected['E5_functional_targets_strict_P1']=['ridge','concat','fv']
    expected['E1_P2_combo']+=['shrunk_additive','shrunk_shared']
    missing=[];checks=[];models=[]
    for task,families in expected.items():
        root=OUT/'runs'/task
        if not (root/'rows.csv').exists():missing.append(task+'/rows.csv');continue
        data=pd.read_csv(root/'rows.csv');train=data.partition.eq('train');val=data.partition.eq('validation');test=data.partition.eq('test')
        assert data['index'].is_unique and train.any() and val.any() and test.any()
        assert data.loc[val].groupby('target_contrast').selection_or_calibration.nunique().eq(1).all()
        if task.startswith('E1_global') or task.startswith('E2_strict') or task=='E5_functional_targets_strict_P1':
            assert not set(data.loc[train,'target_contrast'])&set(data.loc[val|test,'target_contrast'])
        if task.startswith('E5_'):
            assert not set(data.loc[train|val,'target_contrast'])&set(data.loc[test,'target_contrast'])
        if task.startswith('E1_donor'):
            assert data.groupby('donor_id').partition.nunique().eq(1).all()
            assert data.loc[train,'donor_id'].nunique()==2 and data.loc[val,'donor_id'].nunique()==1 and data.loc[test,'donor_id'].nunique()==1
        if task=='E1_external_study_P1':
            assert set(data.loc[train,'donor_id'])=={'CE0008162','CE0006864'}
            assert set(data.loc[test,'donor_id'])=={'K562_cellline','RPE1_cellline'}
        checks.append({'task':task,'train_rows':int(train.sum()),'selection_rows':int((val&data.selection_or_calibration.eq('selection')).sum()),'calibration_rows':int((val&data.selection_or_calibration.eq('calibration')).sum()),'test_rows':int(test.sum()),'test_targets':int(data.loc[test,'target_contrast'].nunique()),'manifest_sha256':sha(root/'rows.csv')})
        for family in families:
            folder=root/family;summary=folder/'summary.json'
            if not summary.exists():missing.append(task+'/'+family);continue
            s=json.loads(summary.read_text())
            if s.get('deterministic'):
                assert s['test']['rows']==int(test.sum())
                if family.startswith('shrunk_'):
                    extra=json.loads((folder/'result.json').read_text())
                    assert not extra['test_used_for_selection']
                    assert extra['selected']==min(extra['choices'],key=lambda r:r['selection_MAE'])
                    f=pd.read_csv(folder/'test_rows.csv');assert set(f['index'])==set(data.loc[test,'index'])
                    assert abs(float(f.groupby(['donor','context']).mae.mean().groupby('donor').mean().mean())-s['test']['mae'])<1e-10
                continue
            assert s['seeds']==[11,29,47] and len(s['runs'])==3
            for seed,result in zip(s['seeds'],s['runs']):
                assert result['fit']['seed']==seed and not result['test_used_for_model_selection']
                seed_root=folder/str(seed);frame=pd.read_csv(seed_root/'test_rows.csv')
                assert set(frame['index'])==set(data.loc[test,'index']) and not frame['index'].duplicated().any()
                score=float(frame.groupby(['donor','context']).mae.mean().groupby('donor').mean().mean())
                assert abs(score-result['test']['mae'])<1e-10
                # Training uses a predeclared 1e-6 minimum improvement, not the
                # absolute numerical minimum when a later gain is below tolerance.
                accepted=float('inf')
                for event in result['fit']['history']:
                    if event['validation_mae']<accepted-1e-6:accepted=event['validation_mae']
                assert abs(accepted-result['fit']['best_selection_mae'])<1e-12
                models.append({'task':task,'family':family,'seed':seed,'checkpoint':str((seed_root/'model.pt').relative_to(OUT)),'sha256':sha(seed_root/'model.pt'),'selected_lr':result['fit']['lr'],'steps':result['fit']['steps'],'test_mae':score})
    if missing:
        write_json(OUT/'completion_audit.json',{'state':'incomplete','missing':missing,'completed_neural_checkpoints':len(models),'checked_tasks':checks});print(json.dumps({'missing':missing},indent=2));return
    # Check cached official labels against independently read source rows.
    cached=np.load(BASE/'training_first/effects.npy',mmap_mode='r');primary=pd.read_csv(PREP/'primary_rows.csv')
    with h5py.File(BASE/'large_data/GWCD4i.DE_stats.h5ad','r') as f:
        loc={v:i for i,v in enumerate(index(f['obs']))};indices=np.random.default_rng(20261026).choice(len(primary),size=32,replace=False)
        for i in indices:np.testing.assert_array_equal(cached[i],np.asarray(f['layers/log_fc'][loc[str(primary.iloc[i]['index'])]],dtype=np.float32))
    protected=json.loads((BASE/'protected_files_before.json').read_text(encoding='utf-8'));root=BASE.parents[2]
    assert all(sha(root/p['path'])==p['sha256'] for p in protected)
    frozen=json.loads((BASE/'prepared/FREEZE_LOCK.json').read_text());assert all(sha(BASE/'prepared'/name)==h for name,h in frozen['sha256'].items())
    csv=pd.DataFrame(models);csv.to_csv(OUT/'analysis/model_checkpoint_manifest.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'completion_audit.json',{'state':'all_registered_computational_runs_complete','task_settings':len(expected),'method_task_settings':sum(map(len,expected.values())),'neural_checkpoints':len(models),'deterministic_baseline_settings':sum(len(v) for v in expected.values())-len(models)//3,'seed_count':3,'fixed_primary_source_rows_independently_verified':32,'protected_original_files_unchanged':len(protected),'original_protocol_hashes_unchanged':True,'checks':checks,'independent_biological_donors_in_main_cohort':4,'scope_gaps':['No fully cell-count/donor/platform matched broad pretraining comparison; E3 source comparison remains confounded','No independent primary-T CRISPRi study with harmonized complete response gold constructed; Replogle tests are non-primary-T OOD','No real organ composition/transport or longitudinal intervention validation; synthetic identity check is not biological evidence','Published upstream curated targets and pooled primary DE feature selection remain limitations','Mechanism regularizer and latent L2 residual shrinkage are tested; no full hierarchical Bayesian model or pretrained SE encoder claim'],'completion_time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()})
    packages={name:md.version(name) for name in ['numpy','pandas','torch','h5py','scipy','scikit-learn','transformers','lightning']}
    sources={p.name:sha(p) for p in (BASE/'scripts').glob('full_suite_*.py')}
    write_json(OUT/'software_and_code_manifest.json',{'python':sys.version,'packages':packages,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'source_code_sha256':sources,'training_precision':'FP32, no autocast','stochastic_repeats':3,'timing_warning':'Selected-fit seconds exclude discarded hyperparameter trials and some cache setup. Family directory wall time includes tuning/evaluation/recovery and concurrent workloads, not exclusive GPU hours. Older fits record process-cumulative peak memory; later fits explicitly reset per-fit peak.','original_inputs_modified':False})
    print(json.dumps({'state':'complete','tasks':len(expected),'neural_models':len(models)},indent=2))
if __name__=='__main__':main()
