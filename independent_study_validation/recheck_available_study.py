"""Complete an independently sourced, already downloaded study audit first.

This is a cell-line shift test, never a substitute for primary-T confirmation.
No training or model selection on the external study is performed.
"""
from evaluate_frozen import *
from full_suite_core import predict,build_model
from full_suite_run import baseline_predictions
X=B/'full_experiments';TASK=X/'runs/E1_external_study_P1';DEST=R/'available_study';DEST.mkdir(exist_ok=True)
def put(n,v):(DEST/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
 plan={'date':'2026-10-27','study':'Replogle et al. 2022, K562/RPE1; independent of primary-T GWT experiment, strongly shifted cell backgrounds','purpose':'Recheck saved held-study outputs and add predefined readouts, positive/negative controls before writing; not a new pristine confirmatory test','selection':'Only original GWT selection donor; no external training or selection; no new fitting','models':['fv','concat','state_official','ridge','zero','training_mean','same_target_mean'],'metrics':['equal-cell-line/target MAE','Pearson delta','direction |gold|>0.1','truth-aligned amplitude','within-cell-line cosine retrieval with ties'],'negative':'20 wrong-target derangements, seed20261027','positive':'guide-disjoint halves when at least two distinct guide assignments available, all control guide assignments split alternately; >=5 cells each target half and control half','uncertainty':'2000 paired target-cluster bootstrap draws, descriptive conditional on these targets; not donor replication','source_retention':'All 6795 shared published retained genes; no outcome-based external gene selection; this is not raw realignment or all-genome coverage','prior_test_exposure':'Original MAE and other existing outputs already examined; additional analyses are exploratory'}
 pp=DEST/'protocol.json'
 if not pp.exists():put('protocol.json',plan)
 else:assert json.loads(pp.read_text())==plan
 m=pd.read_csv(TASK/'rows.csv');y=np.load(X/'external_data/study_holdout_effects.npy',mmap_mode='r');ck=torch.load(TASK/'fv/11/model.pt',map_location='cpu',weights_only=False);g=np.array(ck['gene_ids']);ctrl=np.load(X/'external_data/study_controls.npz');m.attrs['control_override']={tuple(k.split('|')):v for k,v in zip(ctrl['keys'],ctrl['values'])};m.attrs['hashed_context_encoding']=True;f=features(y,m,g,'P1');ii=np.flatnonzero(m.partition.eq('test'));meta=m.iloc[ii].copy().reset_index(drop=True);meta['target']=meta.target_contrast;meta['condition']=meta.culture_condition;true=measured(y,m,ii);allrows=[];predictions={};checks=[];hashes={}
 for family in ['fv','concat','state_official','ridge','zero','training_mean','same_target_mean']:
  ps=[]
  for seed in ([11,29,47] if family in ['fv','concat','state_official'] else [0]):
   if family in ['zero','training_mean','same_target_mean']:p=baseline_predictions(y,m,f,ii,family)
   elif family=='ridge':
    path=TASK/'ridge/ridge_model.npz';r=np.load(path);x=np.concatenate([f['u'][ii],f['z'][ii],np.ones((len(ii),1),np.float32)],1);p=x@r['weights']+r['training_mean'];hashes[str(path.relative_to(B))]=sha(path)
   else:
    path=TASK/family/str(seed)/'model.pt';a=torch.load(path,map_location='cpu',weights_only=False);model=build_model(family,len(g)).cuda();model.load_state_dict(a['state_dict']);p=predict(model,f,ii,a['mean']);del model;torch.cuda.empty_cache();hashes[str(path.relative_to(B))]=sha(path)
    saved=TASK/family/str(seed)/'test_predictions.npy'
    if saved.exists():
     error=float(np.max(np.abs(p-np.load(saved))));assert error<2e-5;checks.append({'family':family,'seed':seed,'max_prediction_difference_from_original':error})
   ps.append(p);q=meta[['target','condition']].copy();q['model']=family;q['seed']=seed
   for c in sorted(meta.condition.unique()):
    ids=np.flatnonzero(meta.condition.eq(c));v=row_metrics(true[ids],p[ids])
    for key,value in v.items():q.loc[ids,key]=value
   old=TASK/family/(str(seed)+'/' if seed else '')/'test_rows.csv'
   if old.exists():
    gold=pd.read_csv(old);error=float(np.max(np.abs(q.mae.to_numpy()-gold.mae.to_numpy())));assert error<2e-6;checks.append({'family':family,'seed':seed,'max_MAE_difference_from_original':error})
   allrows.append(q)
  predictions[family]=np.stack(ps).mean(0);np.save(DEST/(family+'_predictions.npy'),predictions[family])
 for q in allrows:q.attrs={}
 rows=pd.concat(allrows,ignore_index=True);rows.to_csv(DEST/'model_rows.csv',index=False);summary={}
 for fam,frame in rows.groupby('model'):
  vals=[]
  for seed,df in frame.groupby('seed'):vals.append(df.groupby('condition')[['mae','pearson','direction','amplitude','retrieval_top1','retrieval_rank']].mean().mean())
  v=pd.DataFrame(vals).mean();summary[fam]={k:float(v[k]) if np.isfinite(v[k]) else None for k in v.index};summary[fam]['seeds']=len(vals);summary[fam]['by_context_MAE']=frame.groupby('condition').mae.mean().to_dict()
 put('summary.json',summary);put('prediction_recheck.json',checks)
 rng=np.random.default_rng(20261027);neg=[]
 for repeat in range(20):
  for c in sorted(meta.condition.unique()):
   ids=np.flatnonzero(meta.condition.eq(c));perm=rng.permutation(len(ids))
   while (perm==np.arange(len(ids))).any():perm=rng.permutation(len(ids))
   for fam in ['fv','same_target_mean']:
    q=row_metrics(true[ids],predictions[fam][ids[perm]]);neg.append({'repeat':repeat,'condition':c,'model':fam,**{k:float(np.nanmean(v)) if np.isfinite(v).any() else None for k,v in q.items()}})
 pd.DataFrame(neg).to_csv(DEST/'wrong_target_controls.csv',index=False)
 # Raw matrices already exist and are read only; aggregate disjoint guide IDs.
 from run_first_training import h5py,column,index,rows as h5rows
 positive=[];availability=[]
 for name,filename in [('K562','ReplogleWeissman2022_K562_essential.h5ad'),('RPE1','ReplogleWeissman2022_rpe1.h5ad')]:
  path=X/'external_data'/filename
  with h5py.File(path,'r') as h:
   ids=column(h['var']['ensembl_id']).astype(str);ids=np.array([z.split('.')[0] for z in ids]);gm={v:j for j,v in enumerate(ids)};gp=np.array([gm[z] for z in g]);lab=column(h['obs']['perturbation']).astype(str);guide=column(h['obs']['guide_id']).astype(str)
   # Published aggregation preserves unambiguous symbol->source gene mapping.
   ag=np.load(X/'external_data'/(name+'_pseudobulk.npz'));mapping=dict(zip(ag['target_names'].astype(str),ag['target_ids'].astype(str))) if 'target_names' in ag else None
   if mapping is None:
    with h5py.File(B/'large_data/GWCD4i.pseudobulk_merged.h5ad','r') as src:
     ref=pd.DataFrame({'id':index(src['var']),'symbol':column(src['var']['gene_name']).astype(str)}).drop_duplicates()
    unique=ref.groupby('symbol').id.nunique();mapping=dict(zip(ref[ref.symbol.isin(unique[unique==1].index)].symbol,ref[ref.symbol.isin(unique[unique==1].index)].id))
   targets=np.array(['NTC' if v=='control' else mapping.get(v,'UNMAPPED') for v in lab]);wanted=sorted(meta.loc[meta.condition.eq(name),'target'].unique());group=np.full(len(lab),-1,np.int32);used=[]
   for target in wanted+['NTC']:
    where=np.flatnonzero(targets==target);gg=sorted(set(guide[where]));parts=[set(gg[::2]),set(gg[1::2])]
    if len(gg)<2:continue
    n=[int(np.isin(guide[where],list(part)).sum()) for part in parts]
    if min(n)<5:continue
    j=len(used);used.append(target)
    for k,part in enumerate(parts):group[where[np.isin(guide[where],list(part))]]=j*2+k
   nc=np.bincount(group[group>=0],minlength=len(used)*2);counts=np.zeros((len(used)*2,len(ids)),np.float64)
   for start in range(0,len(lab),512):
    end=min(start+512,len(lab));ok=group[start:end]>=0
    if ok.any():
     block=h5rows(h['X'],start,end)
     if not np.isfinite(block).all() or (block<0).any():raise ValueError('Non-count data')
     np.add.at(counts,group[start:end][ok],block[ok])
   availability.append({'cell_line':name,'all_shared_targets':len(wanted),'targets_with_two_guide_halves':len(used)-int('NTC' in used),'guide_halves_are_not_donors':True})
   if 'NTC' not in used:continue
   j=used.index('NTC');control=counts[j*2:j*2+2];control=np.log2(1+control/control.sum(1,keepdims=True)*1e6)
   for j,t in enumerate(used):
    if t=='NTC':continue
    val=counts[j*2:j*2+2];eff=np.log2(1+val/val.sum(1,keepdims=True)*1e6)-control;q=row_metrics(eff[0:1,gp],eff[1:2,gp]);positive.append({'target':t,'condition':name,'left_cells':int(nc[j*2]),'right_cells':int(nc[j*2+1]),**{k:float(v[0]) if np.isfinite(v[0]) else None for k,v in q.items()}})
 pd.DataFrame(positive).to_csv(DEST/'guide_disjoint_positive_controls.csv',index=False);put('positive_control_availability.json',availability)
 avg=rows.groupby(['model','target','condition']).mae.mean().reset_index();wide=avg.groupby(['target','model']).mae.mean().unstack();paired={};rng=np.random.default_rng(20261027)
 for c in ['zero','training_mean','same_target_mean','ridge','concat']:
  d=(wide.fv-wide[c]).dropna().to_numpy();bb=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(2000)]);paired[c]={'FV_minus_comparator_MAE':float(d.mean()),'descriptive_target_bootstrap_95_interval':np.quantile(bb,[.025,.975]).tolist(),'targets':len(d),'not_donor_population_inference':True}
 put('paired_target_contrasts.json',paired);put('completion.json',{'state':'complete','source_study':'Replogle2022','external_cells':'K562 and RPE1, no primary-T confirmation','test_rows':len(meta),'test_targets':int(meta.target.nunique()),'shared_genes':len(g),'source_model_SHA256':hashes,'source_input_schema_and_full_retained_counts_checked':True,'prior_test_exposure':True,'retrained_models':0,'positive_controls':len(positive),'updated_epoch':time.time()});print(json.dumps({'summary':summary,'positive_availability':availability,'paired':paired},indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:put('status.json',{'state':'failed','error':str(e),'traceback':traceback.format_exc()});raise
