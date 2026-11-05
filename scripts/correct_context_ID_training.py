"""Continue train-only warm starts after removing the unregularized ID bypass."""
from completion_core import *
from full_suite_run import load_primary
from completion_metrics import fast_scores as scores
import shutil,subprocess,argparse

def ready(name):
    p=ROOT/(name+'_status.json');return p.exists() and json.loads(p.read_text()).get('state')=='complete'

def finish_family(task,y,m,g,f,dc,donors,family,archive,mask=None,control_source=None):
    dest=task/family;dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():
        old=json.loads((dest/'summary.json').read_text())
        if old.get('identifiability_corrected') or family=='fv_continued':return
        raise RuntimeError('Unexpected non-corrected output remains')
    write_json(ROOT/('corrective_'+os.environ.get('CORRECT_BRANCH','all')+'_progress.json'),{'state':'running','task':task.name,'family':family,'updated_epoch':time.time()})
    results=[];ti=np.flatnonzero(m.partition.eq('test'));base_family='fv' if family=='fv_continued' else family
    for seed in PLAN['seeds']:
        folder=dest/str(seed);folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():results.append(json.loads((folder/'result.json').read_text()));continue
        warm_path=(control_source if family=='fv_continued' else archive)/str(seed)/'model.pt';cp=torch.load(warm_path,weights_only=False,map_location='cpu')
        assert np.array_equal(np.asarray(cp.get('genes',cp.get('gene_ids'))),g)
        np.testing.assert_allclose(cp['mean'],f['mean'],rtol=1e-5,atol=1e-6)
        for k,v in f['transform'].items():np.testing.assert_allclose(cp['feature_transform'][k],v,rtol=1e-5,atol=1e-6)
        lr=cp['fit']['lr'];strength=0. if family=='fv_continued' else cp['fit']['strength']
        proxy=ObservedArray(y,mask) if mask is not None else y
        model,info=fit_new(proxy,m,f,base_family,seed,lr,strength,folder,dc,donors,mask,warm_checkpoint=cp,max_steps=1000)
        pred=prediction(model,f,ti,dc);test=scores(y,m,ti,pred,folder/'test_rows.csv')
        if isinstance(model,IdentifiableHierarchy):assert bool((model.state.weight[:,32:]==0).all())
        state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        torch.save({'state_dict':state,'family':base_family,'genes':g.tolist(),'permission':cp['permission'],'mean':f['mean'],'feature_transform':f['transform'],'train_donor_mapping':{d:i for i,d in enumerate(sorted(m.loc[m.partition.eq('train'),'donor_id'].unique()))} if 'donor_id' in m else {'pooled':0},'protocol_sha256':sha(ROOT/'protocol.json'),'correction_protocol_sha256':sha(ROOT/'identifiability_correction_protocol.json'),'warm_start_sha256':sha(warm_path),'fit':info},folder/'model.pt')
        if task.name=='hierarchical_functional_P1':np.save(folder/'test_predictions.npy',pred)
        value={'fit':info,'warm_start_fit':cp['fit'],'warm_start_sha256':sha(warm_path),'test':test,'test_used_for_selection':False,'tuning_trials':[info],'correction':'one-hot context path removed for hierarchical models; matched 1000-step continuation for FV'}
        write_json(folder/'result.json',value);results.append(value);del model,cp,pred,state;gc.collect();torch.cuda.empty_cache()
    write_json(dest/'summary.json',{'seeds':PLAN['seeds'],'mean_test_MAE':float(np.mean([r['test']['mae'] for r in results])),'sd_test_MAE':float(np.std([r['test']['mae'] for r in results],ddof=1)),'runs':results,'identifiability_corrected':family!='fv_continued','additional_continuation_steps_max':1000})

def main(branch):
    os.environ['CORRECT_BRANCH']=branch
    while not all(ready(n) for n in ('hierarchy','pooled','missing')):
        if not ready('hierarchy') and not (ROOT/'hierarchy.lock').exists():raise RuntimeError('Hierarchy worker stopped before success')
        if not ready('pooled') and not (ROOT/'pooled.lock').exists():raise RuntimeError('Pooled worker stopped before success')
        time.sleep(5)
    archive_root=(ROOT/'archive_pre_identifiability_fix').resolve();archive_root.mkdir(exist_ok=True)
    registry=json.loads((ROOT/'task_registry.json').read_text());primary_y,primary_m,primary_g=load_primary();donor_y=np.load(PREP/'donor_effects.npy',mmap_mode='r');donor_g=pd.read_csv(PREP/'donor_genes.csv').gene_id.to_numpy()
    tasks=[t for t,fs in registry['tasks'].items() if any(n in fs for n in ('hierarchical','hierarchical_pooled'))]
    if branch=='donor':tasks=[t for t in tasks if t.startswith('hierarchical_donor')]
    elif branch=='pooled':tasks=[t for t in tasks if not t.startswith('hierarchical_donor')]
    for name in tasks:
        task=ROOT/'runs'/name;m=pd.read_csv(task/'rows.csv');is_donor=name.startswith('hierarchical_donor');y,g=(donor_y,donor_g) if is_donor else (primary_y,primary_g)
        mask=make_mask(y,m,int(name.split('_')[1])/100) if name.startswith('missing_') else None
        if mask is not None:_,f=mask_features(y,m,g,mask)
        else:f=features(y,m,g,'P2' if name=='hierarchical_combo_P2' else 'P1')
        train=m.partition.eq('train');donors=sorted(m.loc[train,'donor_id'].unique()) if is_donor else ['pooled'];dc=m.donor_id.map({d:i for i,d in enumerate(donors)}).fillna(-1).to_numpy(np.int64) if is_donor else np.zeros(len(m),np.int64)
        families=[n for n in registry['tasks'][name] if n in ('hierarchical','fixed_gaussian','unshrunk_offsets','hierarchical_pooled')]
        for family in families:
            source=(task/family).resolve();archive=(archive_root/name/family).resolve()
            assert source.is_relative_to(ROOT.resolve()) and archive.is_relative_to(ROOT.resolve())
            if source.exists() and (source/'summary.json').exists() and not json.loads((source/'summary.json').read_text()).get('identifiability_corrected'):
                if archive.exists():raise RuntimeError('Archive already exists; inspect before moving')
                archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),str(archive))
            assert archive.exists()
            finish_family(task,y,m,g,f,dc,len(donors),family,archive,mask)
        if name.startswith('hierarchical_donor'):
            fold=int(name[-1]);old='E1_donor_P1' if fold==0 else 'E1_donor_P1_fold'+str(fold);control=OUT/'runs'/old/'fv'
        elif name.startswith('missing_'):control=task/'fv'
        else:control=OUT/'runs'/({'hierarchical_combo_P2':'E1_P2_combo','hierarchical_global_P1':'E1_global_target_P1','hierarchical_functional_P1':'E5_functional_targets_strict_P1'}[name])/'fv'
        finish_family(task,y,m,g,f,dc,len(donors),'fv_continued',None,mask,control)
        del f,mask;gc.collect();torch.cuda.empty_cache()
    write_json(ROOT/('corrective_'+branch+'_status.json'),{'state':'complete','updated_epoch':time.time()})
    if branch!='all':return
    for key,script in [('calibration','calibrate_training_completion.py'),('auxiliary','evaluate_completion_auxiliary.py'),('audit','analyse_training_completion.py')]:
        with (ROOT/(key+'.stdout.log')).open('ab') as out,(ROOT/(key+'.stderr.log')).open('ab') as err:
            p=subprocess.Popen([sys.executable,'-B','-u',str(BASE/'scripts'/script)],cwd=BASE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);(ROOT/(key+'.pid')).write_text(str(p.pid));code=p.wait()
        if code:raise RuntimeError(key+' failed')
    assert json.loads((ROOT/'audit.json').read_text())['state']=='all_registered_additional_runs_complete'
    write_json(ROOT/'evaluation_status.json',{'state':'complete','updated_epoch':time.time()})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['all','donor','pooled'],default='all');args=parser.parse_args()
    if args.branch=='all':assert not any((ROOT/('corrective_'+b+'.lock')).exists() for b in ('donor','pooled')),'Sharded workers already running'
    else:assert not (ROOT/'corrective.lock').exists(),'Full corrective worker already running'
    lock=ROOT/('corrective.lock' if args.branch=='all' else 'corrective_'+args.branch+'.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main(args.branch)
    finally:lock.unlink(missing_ok=True)
