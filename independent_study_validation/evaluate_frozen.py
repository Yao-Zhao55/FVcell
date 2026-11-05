"""Read-only transfer of frozen source models to a complete external study."""
from pathlib import Path
import sys,json,re,time,hashlib,os,traceback
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'scripts'))
from full_suite_core import np,pd,torch,features,Factorized,Concatenation,measured,sha,hashnum
from completion_core import IdentifiableHierarchy
P=R/'prepared';O=R/'results';O.mkdir(exist_ok=True)
def save(name,value):
 (O/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def state(stage,**kw):save('status.json',{'stage':stage,'pid':os.getpid(),'updated_epoch':time.time(),**kw})
def row_metrics(true,pred):
 t=true.astype(np.float64);p=pred.astype(np.float64);a=t-t.mean(1,keepdims=True);b=p-p.mean(1,keepdims=True);den=np.sqrt((a*a).sum(1)*(b*b).sum(1));nz=den>0
 cor=np.full(len(t),np.nan);cor[nz]=(a[nz]*b[nz]).sum(1)/den[nz]
 sel=np.abs(t)>.1;n=sel.sum(1);direction=np.divide(((np.sign(t)==np.sign(p))&sel).sum(1),n,out=np.full(len(t),np.nan),where=n>0)
 energy=(t*t).sum(1);amplitude=np.divide((t*p).sum(1),energy,out=np.full(len(t),np.nan),where=energy>0)
 norm=np.sqrt((p*p).sum(1));pn=np.divide(p,norm[:,None],out=np.zeros_like(p),where=norm[:,None]>0)
 norm=np.sqrt((t*t).sum(1));tn=np.divide(t,norm[:,None],out=np.zeros_like(t),where=norm[:,None]>0)
 sim=pn@tn.T;diag=sim.diagonal();better=(sim>diag[:,None]+1e-10).sum(1);ties=(np.abs(sim-diag[:,None])<=1e-10).sum(1)
 top=np.where(better==0,1/np.maximum(ties,1),0)
 return {'mae':np.abs(t-p).mean(1),'pearson':cor,'direction':direction,'amplitude':amplitude,'retrieval_top1':top,'retrieval_rank':better+(ties+1)/2,'direction_genes':n}
def summarize(frame):
 out={}
 for scope,tags in [('Teff',['Resting_Teff','Stimulated_Teff']),('Treg',['Resting_Treg','Stimulated_Treg']),('all',list(frame.condition.unique()))]:
  f=frame[frame.condition.isin(tags)];out[scope]={k:float(f.groupby('condition')[k].mean().mean()) if f[k].notna().any() else None for k in ['mae','pearson','direction','amplitude','retrieval_top1','retrieval_rank']};out[scope]['rows']=len(f)
 return out
def raw_inputs(targets,contexts,control,model_genes,source_donors):
 mech=np.load(B/'full_experiments/prepared/mechanism.npz');lookup={v:i for i,v in enumerate(mech['target_ids'])};u=np.zeros((len(targets),65),np.float32)
 for i,t in enumerate(targets):
  u[i,:32]=mech['embedding'][lookup[t]] if t in lookup else np.random.default_rng(hashnum(t)).normal(size=32)/np.sqrt(32)
 cp=np.load(B/'full_experiments/prepared/controls.npz');gmap={g:i for i,g in enumerate(cp['gene_ids'])};gp=np.array([gmap[g] for g in model_genes]);cmap={str(c):j for j,c in enumerate(cp['contexts'])};dmap={str(d):j for j,d in enumerate(cp['donors'])}
 projection=np.random.default_rng(20261026).normal(size=(len(model_genes),32)).astype(np.float32)/np.sqrt(len(model_genes))
 z=np.zeros((len(targets),35),np.float32)
 for i,c in enumerate(contexts):
  source=np.stack([cp['control_log2cpm'][dmap[d]*3+cmap[c],gp] for d in source_donors]).mean(0)
  valid=np.isfinite(control[i]);source[valid]=control[i,valid];z[i,:32]=source@projection;z[i,32+cmap[c]]=1
 return u,z,np.array([cmap[c] for c in contexts],np.int64)
def transformed(u,z,tr):return (u-tr['u_mean'])/tr['u_scale'],(z-tr['z_mean'])/tr['z_scale']
def checkpoint_predictions(path,targets,contexts,ctrl,expected_genes):
 ck=torch.load(path,map_location='cpu',weights_only=False);genes=ck.get('genes',ck.get('gene_ids'));assert genes==list(expected_genes)
 donors=ck.get('train_donor_mapping',{'CE0006864':0,'CE0008162':1});u,z,c=raw_inputs(targets,contexts,ctrl,genes,sorted(donors));u,z=transformed(u,z,ck['feature_transform']);family=ck['family']
 if family in ['hierarchical','fixed_gaussian']:model=IdentifiableHierarchy(len(genes),len(donors),family)
 elif family=='concat':model=Concatenation(len(genes))
 else:model=Factorized(len(genes))
 model.load_state_dict(ck['state_dict']);model.eval();model=model.cuda();tt=[torch.tensor(a,device='cuda') for a in [u,z,c]]
 with torch.no_grad():
  val=model(*tt,torch.full((len(u),),-1,device='cuda',dtype=torch.long))[0] if isinstance(model,IdentifiableHierarchy) else model(*tt)[0]
  pred=val.cpu().numpy()+ck['mean']
 del model;torch.cuda.empty_cache();return pred,ck
def main():
 proto=json.loads((R/'protocol_frozen.json').read_text(encoding='utf-8'));assert sha(R/'protocol_frozen.json')==json.loads((R/'FREEZE_LOCK.json').read_text())['protocol_sha256']
 for name,value in proto['frozen_files'].items():assert sha(B/name)==value,'Frozen source changed: '+name
 a=np.load(P/'guide_pseudobulk.npz');counts=a['counts'];nc=a['n_cells'];guides=a['guides'].astype(str);eg=a['gene_ids'].astype(str)
 genes=pd.read_csv(B/'full_experiments/prepared/donor_genes.csv').gene_id.astype(str).to_numpy();emap={g:i for i,g in enumerate(eg)};mask=np.array([g in emap for g in genes]);sp=np.flatnonzero(mask);ep=np.array([emap[g] for g in genes[mask]])
 if mask.sum()<1000:raise ValueError('Insufficient output gene overlap')
 sys.path.insert(0,str(B/'vendor'))
 from run_first_training import h5py,index,column
 with h5py.File(B/'large_data/GWCD4i.pseudobulk_merged.h5ad','r') as f:
  ref=pd.DataFrame({'id':index(f['var']),'name':column(f['var']['gene_name']).astype(str)})
 ref=ref.drop_duplicates();unique=ref.groupby('name').id.nunique();name2id={r['name']:r['id'] for r in ref[ref.name.isin(unique[unique==1].index)].to_dict('records')}
 syms=np.array([re.sub(r'_\d+_CRISPRi$','',g) for g in guides]);ntc=np.char.startswith(syms.astype(str),'Non-Targeting');symbols=sorted(set(syms[~ntc]));tags=['Resting_Treg','Stimulated_Treg','Resting_Teff','Stimulated_Teff'];rows=[];truth=[];controls=[];positive=[]
 for k,tag in enumerate(tags):
  cc=counts[ntc,k].sum(0);nn=nc[ntc,k].sum()
  if nn<200:raise ValueError('Too few controls '+tag)
  ctl=np.log2(1+cc/cc.sum()*1e6);full=np.full(len(genes),np.nan,np.float32);full[sp]=ctl[ep]
  for sym in symbols:
   inds=np.flatnonzero(syms==sym);n=int(nc[inds,k].sum());target=name2id.get(sym)
   if n<50 or target is None:continue
   c=counts[inds,k].sum(0);effect=(np.log2(1+c/c.sum()*1e6)-ctl)[ep].astype(np.float32)
   rows.append({'target':target,'symbol':sym,'condition':tag,'source_context':'Rest' if k%2==0 else 'Stim48hr','n_cells':n,'n_control_cells':int(nn),'n_guides':len(inds)});truth.append(effect);controls.append(full.copy())
   # This tests guide agreement, not biological donor replication.
   left=inds[::2];right=inds[1::2];ci=np.flatnonzero(ntc);ni=[ci[::2],ci[1::2]]
   if len(right) and min(nc[left,k].sum(),nc[right,k].sum())>=25 and min(nc[ni[0],k].sum(),nc[ni[1],k].sum())>=100:
    values=[]
    for ii,jj in [(left,ni[0]),(right,ni[1])]:
     x=counts[ii,k].sum(0);z=counts[jj,k].sum(0);values.append((np.log2(1+x/x.sum()*1e6)-np.log2(1+z/z.sum()*1e6))[ep])
    q=row_metrics(values[0][None,:],values[1][None,:]);positive.append({'target':target,'symbol':sym,'condition':tag,**{a:float(v[0]) if np.isfinite(v[0]) else None for a,v in q.items()}})
 meta=pd.DataFrame(rows);true=np.stack(truth);ctrl=np.stack(controls);targets=meta.target.astype(str).to_numpy();contexts=meta.source_context.to_numpy();meta.to_csv(P/'evaluation_rows.csv',index=False);np.save(P/'gold.npy',true);np.save(P/'output_gene_mask.npy',mask);np.savez_compressed(P/'external_controls.npz',values=ctrl,gene_ids=np.asarray(genes,dtype=str))
 coverage={'targets':int(meta.target.nunique()),'rows':len(meta),'output_genes':int(mask.sum()),'source_model_genes':len(genes),'missing_control_input_genes':genes[~mask].tolist(),'conditions':meta.groupby('condition').size().to_dict(),'controls_allowed':True,'external_response_features_allowed':False,'donor_identities_verified_disjoint':False,'laboratories_overlap':True,'intervention':'CRISPRi','study_version':'GSE278572 public matrix 2024-09-28','frozen_source_files_verified':len(proto['frozen_files'])};save('coverage.json',coverage)
 frames=[];preds={};source_ck=None
 for fam in ['hierarchical','fixed_gaussian','fv_continued','concat_original_P1']:
  pp=[]
  for seed in [11,29,47]:
   path=B/f'training_completion/runs/hierarchical_donor_fold0/{fam}/{seed}/model.pt' if fam!='concat_original_P1' else B/f'full_experiments/runs/E1_donor_P1/concat/{seed}/model.pt'
   pred,ck=checkpoint_predictions(path,targets,contexts,ctrl,genes);source_ck=ck if source_ck is None else source_ck;pred=pred[:,sp];pp.append(pred);q=meta.copy();q['model']=fam;q['seed']=seed
   for k in range(4):
    ii=np.flatnonzero(meta.condition.eq(tags[k]));v=row_metrics(true[ii],pred[ii])
    for metric,val in v.items():q.loc[ii,metric]=val
   frames.append(q);state('predicting',model=fam,seed=seed)
  preds[fam]=np.stack(pp).mean(0);np.save(O/(fam+'_predictions_seed_mean.npy'),preds[fam])
 # Deterministic source baselines are fitted only from original training data.
 sm=pd.read_csv(B/'full_experiments/runs/E1_donor_P1/rows.csv');sy=np.load(B/'full_experiments/prepared/donor_effects.npy',mmap_mode='r');state('source_only_baselines')
 f=features(sy,sm,genes,'P1');source_mean=f['mean'];tm={t:i for i,t in enumerate(f['target_ids'])};targetmeans=[]
 for t in targets:
  j=tm.get(t);targetmeans.append(f['target_sums'][j]/f['target_counts'][j] if j is not None and f['target_counts'][j] else source_mean)
 shared=np.stack(targetmeans);basepred={'zero':np.zeros_like(true),'source_training_mean':np.tile(source_mean[sp],(len(meta),1)),'source_same_target_mean':shared[:,sp]}
 rid=np.load(B/'full_experiments/runs/E1_donor_P1/ridge/ridge_model.npz');u,z,c=raw_inputs(targets,contexts,ctrl,genes,['CE0006864','CE0008162']);u,z=transformed(u,z,rid);x=np.concatenate([u,z,np.ones((len(u),1),np.float32)],1);basepred['source_ridge_P1']=(x@rid['weights']+rid['training_mean'])[:,sp]
 # Same 4x4 shrinkage grid as the prior source-only additive family.
 tr=np.flatnonzero(sm.partition.eq('train'));vi=np.flatnonzero(sm.partition.eq('validation')&sm.selection_or_calibration.eq('selection'));programs=np.zeros((3,len(genes)),np.float64);nums=np.zeros(3,np.int64);cm={'Rest':0,'Stim48hr':1,'Stim8hr':2};source_codes=sm.culture_condition.map(cm).to_numpy()
 for start in range(0,len(tr),128):
  ii=tr[start:start+128];val=measured(sy,sm,ii);tc=f['target_codes'][ii];n=f['target_counts'][tc];leave=np.where((n>1)[:,None],(f['target_sums'][tc]-val)/np.maximum(n[:,None]-1,1),source_mean)
  for k in range(3):
   ok=source_codes[ii]==k;programs[k]+=(val[ok]-leave[ok]).sum(0,dtype=np.float64);nums[k]+=ok.sum()
 programs=(programs/nums[:,None]).astype(np.float32);grid=[(a,b) for a in [0.,.25,.5,1.] for b in [0.,.25,.5,1.]];errors=np.zeros((len(grid),3),np.float64);vn=np.zeros(3,np.int64)
 for start in range(0,len(vi),64):
  ii=vi[start:start+64];tt=torch.tensor(measured(sy,sm,ii),device='cuda');tc=f['target_codes'][ii];n=f['target_counts'][tc];ss=np.where((n>0)[:,None],f['target_sums'][tc]/np.maximum(n[:,None],1),source_mean);ss=torch.tensor(ss,device='cuda');pr=torch.tensor(programs[source_codes[ii]],device='cuda');mean=torch.tensor(source_mean,device='cuda')
  for j,(a,b) in enumerate(grid):
   err=torch.abs(a*ss+(1-a)*mean+b*pr-tt).mean(1).cpu().numpy()
   for k in range(3):errors[j,k]+=err[source_codes[ii]==k].sum()
  for k in range(3):vn[k]+=(source_codes[ii]==k).sum()
 vals=(errors/vn).mean(1);best=int(np.argmin(vals));a,b=grid[best];basepred['source_shrunk_additive']=(a*shared+(1-a)*source_mean+b*programs[[cm[c] for c in contexts]])[:,sp]
 save('source_additive_selection.json',{'grid':[{'shared_weight':a,'context_weight':b,'source_validation_MAE':float(v)} for (a,b),v in zip(grid,vals)],'selected':{'shared_weight':a,'context_weight':b},'external_outcomes_used_for_selection':False})
 for family,pred in basepred.items():
  preds[family]=pred;np.save(O/(family+'_predictions.npy'),pred);q=meta.copy();q['model']=family;q['seed']=0
  for tag in tags:
   ii=np.flatnonzero(meta.condition.eq(tag));v=row_metrics(true[ii],pred[ii])
   for k,val in v.items():q.loc[ii,k]=val
  frames.append(q)
 frame=pd.concat(frames,ignore_index=True);frame.to_csv(O/'all_model_rows.csv',index=False);summary={}
 for family,q in frame.groupby('model'):
  perseed=[summarize(z) for seed,z in q.groupby('seed')];summary[family]={scope:{metric:float(np.mean([v[scope][metric] for v in perseed if v[scope][metric] is not None])) if any(v[scope][metric] is not None for v in perseed) else None for metric in ['mae','pearson','direction','amplitude','retrieval_top1','retrieval_rank']} for scope in ['Teff','Treg','all']};summary[family]['number_of_seeds']=len(perseed)
 save('summary.json',summary);pd.DataFrame(positive).to_csv(O/'guide_disjoint_positive_controls.csv',index=False)
 # Wrong-target negatives operate only on saved predictions; no fitting.
 rng=np.random.default_rng(20261027);negative=[]
 for repeat in range(20):
  for tag in tags:
   ii=np.flatnonzero(meta.condition.eq(tag));perm=np.arange(len(ii))
   for attempt in range(100):
    rng.shuffle(perm)
    if np.all(perm!=np.arange(len(ii))):break
   if (perm==np.arange(len(ii))).any():raise ValueError('Failed derangement')
   for fam in ['hierarchical','source_same_target_mean']:
    v=row_metrics(true[ii],preds[fam][ii[perm]]);negative.append({'repeat':repeat,'condition':tag,'model':fam,**{k:float(np.nanmean(a)) if np.isfinite(a).any() else None for k,a in v.items()}})
 pd.DataFrame(negative).to_csv(O/'wrong_target_controls.csv',index=False)
 # Descriptive target clusters; preserve each target's contexts and source seeds.
 q=frame.groupby(['model','target','condition']).mae.mean().reset_index();primary=q[q.condition.isin(tags[2:])];wide=primary.groupby(['target','model']).mae.mean().unstack();paired={};rng=np.random.default_rng(20261027)
 for comp in ['zero','source_same_target_mean','source_shrunk_additive','fv_continued','concat_original_P1','fixed_gaussian']:
  delta=(wide['hierarchical']-wide[comp]).dropna().to_numpy();boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
  paired[comp]={'hierarchical_minus_comparator_MAE':float(delta.mean()),'target_bootstrap_descriptive_95_interval':np.quantile(boot,[.025,.975]).tolist(),'targets':len(delta),'not_donor_population_inference':True}
 save('paired_target_contrasts.json',paired);state('complete',coverage=coverage,models=list(summary));print(json.dumps({'coverage':coverage,'summary':summary,'paired':paired},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state('failed',error=str(e),traceback=traceback.format_exc());raise
