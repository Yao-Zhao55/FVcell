"""Frozen-source, post-hoc BCL9 illustration; no model fitting."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parent;B=R.parent
sys.path[:0]=[str(B/'vendor_ml'),str(B/'vendor'),str(B/'scripts'),str(B/'independent_study_validation')]
import numpy as np,pandas as pd
from evaluate_frozen import checkpoint_predictions
from run_first_training import h5py,index,column

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 proto=json.loads((R/'protocol_case_study.json').read_text());assert sha(R/'protocol_case_study.json')==json.loads((R/'CASE_LOCK.json').read_text())['protocol_sha256']
 with h5py.File(B/'large_data/GWCD4i.pseudobulk_merged.h5ad','r') as f:ref=pd.DataFrame({'gene_id':index(f['var']),'symbol':column(f['var']['gene_name']).astype(str)})
 assert ref.symbol.is_unique
 genes=pd.read_csv(B/'full_experiments/prepared/donor_genes.csv').gene_id.astype(str).to_numpy();positions={g:i for i,g in enumerate(genes)};symbol2id=dict(zip(ref.symbol,ref.gene_id))
 allrows=pd.read_csv(B/'full_experiments/runs/E1_donor_P1/rows.csv');ids=[symbol2id[s] for s in proto['targets']];meta=allrows[allrows.target_contrast.isin(ids)].copy();meta['target_symbol']=meta.target_contrast.map(dict(zip(ids,proto['targets'])))
 assert len(meta)==24;raw=np.load(B/'full_experiments/prepared/donor_effects.npy',mmap_mode='r');observed=np.asarray(raw[meta.storage_row.to_numpy()]);assert np.isfinite(observed).all()
 coverage=[];memberrows=[];modules=proto['modules']
 for name,info in modules.items():
  for symbol in info['genes']:coverage.append({'module':name,'symbol':symbol,'gene_id':symbol2id.get(symbol),'present':symbol2id.get(symbol) in positions,'hypothesized_sign':info['hypothesized_sign']})
 for symbol in proto['glioma_coverage_only']+['TCF4','TCF7L2','STAT1','STAT4']:coverage.append({'module':'coverage_only_not_activity_validation','symbol':symbol,'gene_id':symbol2id.get(symbol),'present':symbol2id.get(symbol) in positions,'hypothesized_sign':None})
 cov=pd.DataFrame(coverage);cov.to_csv(R/'gene_coverage.csv',index=False);moduleinds={name:[positions[symbol2id[s]] for s in spec['genes'] if symbol2id.get(s) in positions] for name,spec in modules.items()}
 for i,row in meta.reset_index(drop=True).iterrows():
  for name,info in modules.items():
   for symbol in info['genes']:
    if symbol2id.get(symbol) in positions:memberrows.append({**row.to_dict(),'module':name,'output_symbol':symbol,'observed_response':float(observed[i,positions[symbol2id[symbol]]])})
 pd.DataFrame(memberrows).to_csv(R/'all_observed_member_responses.csv',index=False)
 om=[]
 for name,inds in moduleinds.items():
  for i,row in meta.reset_index(drop=True).iterrows():om.append({**row.to_dict(),'module':name,'n_member_genes':len(inds),'observed_module_mean':float(observed[i,inds].mean())})
 observedmodule=pd.DataFrame(om);observedmodule.to_csv(R/'all_observed_modules.csv',index=False)
 test=meta[meta.partition.eq('test')].reset_index(drop=True);assert len(test)==6 and set(test.donor_id)=={'CE0008678'}
 true=np.asarray(raw[test.storage_row.to_numpy()]);target=test.target_contrast.astype(str).to_numpy();context=test.culture_condition.to_numpy()
 cp=np.load(B/'full_experiments/prepared/controls.npz');gm={g:i for i,g in enumerate(cp['gene_ids'])};gp=np.array([gm[g] for g in genes]);dm={str(v):i for i,v in enumerate(cp['donors'])};cm={str(v):i for i,v in enumerate(cp['contexts'])}
 control=np.stack([cp['control_log2cpm'][dm[d]*3+cm[c],gp] for d,c in zip(test.donor_id,context)]);frames=[];preds={};checkpoints={};max_recheck=0
 for family in ['hierarchical','fixed_gaussian','fv_continued','concat_original_P1']:
  values=[]
  for seed in [11,29,47]:
   model=B/f'training_completion/runs/hierarchical_donor_fold0/{family}/{seed}/model.pt' if family!='concat_original_P1' else B/f'full_experiments/runs/E1_donor_P1/concat/{seed}/model.pt'
   prediction,ck=checkpoint_predictions(model,target,context,control,genes);checkpoints[model.relative_to(B).as_posix()]=sha(model);values.append(prediction)
   existing=pd.read_csv(model.parent/'test_rows.csv').set_index('index')
   for i,row in test.iterrows():
    diff=abs(float(np.abs(prediction[i]-true[i]).mean())-float(existing.loc[row['index'],'mae']));max_recheck=max(max_recheck,diff)
   for name,inds in moduleinds.items():
    for i,row in test.iterrows():frames.append({**row.to_dict(),'model':family,'seed':seed,'module':name,'full_gene_MAE':float(np.abs(prediction[i]-true[i]).mean()),'module_gene_MAE':float(np.abs(prediction[i,inds]-true[i,inds]).mean()),'observed_module_mean':float(true[i,inds].mean()),'predicted_module_mean':float(prediction[i,inds].mean())})
  preds[family]=np.stack(values).mean(0)
 trainingmean=ck['mean'];train=allrows[allrows.partition.eq('train')];targetmeans=np.stack([np.asarray(raw[train[train.target_contrast.eq(t)].storage_row.to_numpy()]).mean(0) for t in target])
 for family,prediction in [('zero',np.zeros_like(true)),('source_training_mean',np.tile(trainingmean,(len(true),1))),('source_same_target_mean',targetmeans)]:
  preds[family]=prediction
  for name,inds in moduleinds.items():
   for i,row in test.iterrows():frames.append({**row.to_dict(),'model':family,'seed':0,'module':name,'full_gene_MAE':float(np.abs(prediction[i]-true[i]).mean()),'module_gene_MAE':float(np.abs(prediction[i,inds]-true[i,inds]).mean()),'observed_module_mean':float(true[i,inds].mean()),'predicted_module_mean':float(prediction[i,inds].mean())})
 assert max_recheck<2e-6,('Source prediction reconstruction mismatch',max_recheck)
 results=pd.DataFrame(frames);results.to_csv(R/'all_case_model_seed_scores.csv',index=False)
 summary=results.groupby(['model','module'])[['full_gene_MAE','module_gene_MAE','observed_module_mean','predicted_module_mean']].mean().reset_index();summary.to_csv(R/'case_model_summary.csv',index=False)
 # Preserve every target, donor, context and member; no top-example selection.
 status={'state':'complete','type':proto['study_type'],'observed_aggregates':24,'targets':2,'source_donors':4,'cultures':3,'held_donor_aggregates':6,'held_donor':'CE0008678','member_coverage':{k:len(v) for k,v in moduleinds.items()},'frozen_checkpoints':checkpoints,'max_full_gene_MAE_reconstruction_difference':max_recheck,'no_model_refit':True,'no_external_paper_raw_data_used':True,'not_mouse_double_KO_or_drug_validation':True,'not_new_independent_study':True,'no_confirmatory_significance_claim':True,'source_protocol_sha256':sha(R/'protocol_case_study.json')}
 (R/'case_completion.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 print(summary.to_string(index=False));print(observedmodule.groupby(['module','target_symbol','culture_condition']).observed_module_mean.agg(['mean','min','max']).to_string());print(json.dumps(status,indent=2))
if __name__=='__main__':main()
