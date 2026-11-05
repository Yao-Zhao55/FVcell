"""Audit completed external data and predictions without fitting or selecting models."""
from pathlib import Path
import sys,json,hashlib,re,time
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'vendor_ml'))
import numpy as np,pandas as pd
from evaluate_frozen import row_metrics

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while a:=f.read(8*1024*1024):h.update(a)
 return h.hexdigest()

def main():
 p=R/'prepared';o=R/'results';proto=read(R/'protocol_frozen.json');lock=read(R/'FREEZE_LOCK.json')
 assert sha(R/'protocol_frozen.json')==lock['protocol_sha256']
 assert read(o/'status.json')['stage']=='complete'
 provenance=read(p/'count_provenance.json');assert provenance['gzip_CRC_verified']
 for file,digest in proto['frozen_files'].items():assert sha(B/file)==digest,file
 transfer=read(R/'transfer_verification.json');assert transfer['received_range_hashes_match_disk']
 assert sha(R/'data/GSE278572_matrix.mtx.gz')==transfer['sha256']==provenance['source_sha256']
 a=np.load(p/'guide_pseudobulk.npz',allow_pickle=False);counts=a['counts'];nc=a['n_cells'];guides=a['guides'];ids=a['gene_ids']
 assert counts.shape==(65,4,36601) and counts.dtype==np.int64 and counts.min()>=0 and len(set(ids))==len(ids)
 assert nc.sum()==provenance['QC_cells']==122538
 features=pd.read_csv(R/'sources/GSE278572_features.tsv.gz',sep='\t',header=None,names=['id','name','type'])
 assert features.id.is_unique and np.array_equal(features[features.type.eq('Gene Expression')].id.to_numpy(),ids)
 assert [features.loc[features.id.eq('HTO_'+str(k)),'name'].item() for k in range(1,5)]==['Resting_Treg','Stimulated_Treg','Resting_Teff','Stimulated_Teff']
 sourcegenes=pd.read_csv(B/'full_experiments/prepared/donor_genes.csv').gene_id.astype(str).to_numpy()
 mask=np.load(p/'output_gene_mask.npy');assert np.array_equal(mask,np.isin(sourcegenes,ids))
 emap=dict(zip(ids,range(len(ids))));ep=np.array([emap[g] for g in sourcegenes[mask]])
 ctrl=np.load(p/'external_controls.npz')['values'];truth=np.load(p/'gold.npy');meta=pd.read_csv(p/'evaluation_rows.csv')
 assert truth.shape==(112,18103) and np.isfinite(truth).all() and (~mask).sum()==26
 assert np.isnan(ctrl[:,~mask]).all() and np.isfinite(ctrl[:,mask]).all()
 tags=['Resting_Treg','Stimulated_Treg','Resting_Teff','Stimulated_Teff'];symbols=np.array([re.sub(r'_\d+_CRISPRi$','',g) for g in guides]);ntc=np.char.startswith(symbols,'Non-Targeting')
 gold_error=0.;control_error=0.
 for i,row in meta.iterrows():
  k=tags.index(row.condition);ii=symbols==row.symbol;c=counts[ii,k].sum(0);n=counts[ntc,k].sum(0)
  baseline=np.log2(1+n/n.sum()*1e6);expected=np.log2(1+c/c.sum()*1e6)-baseline
  gold_error=max(gold_error,float(np.abs(expected[ep]-truth[i]).max()))
  control_error=max(control_error,float(np.abs(baseline[ep]-ctrl[i,mask]).max()))
  assert nc[ii,k].sum()==row.n_cells and nc[ntc,k].sum()==row.n_control_cells
 assert gold_error<1e-6 and control_error<1e-6
 frame=pd.read_csv(o/'all_model_rows.csv');assert len(frame)==1904
 summary=read(o/'summary.json');percondition=frame.groupby(['model','condition']).mean(numeric_only=True).reset_index()
 percondition.to_csv(o/'per_condition_scores.csv',index=False)
 perseed=frame.groupby(['model','seed','condition']).mean(numeric_only=True).reset_index()
 perseed.to_csv(o/'per_seed_condition_scores.csv',index=False)
 for family,scores in summary.items():
  f=frame[frame.model.eq(family)]
  assert sorted(f.seed.unique())==([11,29,47] if scores['number_of_seeds']==3 else [0])
  for scope,conditions in [('Teff',tags[2:]),('Treg',tags[:2]),('all',tags)]:
   for metric in ['mae','pearson','direction','amplitude','retrieval_top1','retrieval_rank']:
    v=f[f.condition.isin(conditions)].groupby(['seed','condition'])[metric].mean().groupby('seed').mean().mean()
    assert (not np.isfinite(v) and scores[scope][metric] is None) or abs(v-scores[scope][metric])<1e-10
 assert all(meta.groupby('condition').target.nunique()==28)
 aligned=[]
 for family in ['hierarchical','source_same_target_mean']:
  file=family+('_predictions_seed_mean.npy' if family=='hierarchical' else '_predictions.npy')
  prediction=np.load(o/file);assert prediction.shape==truth.shape and np.isfinite(prediction).all()
  for tag in tags:
   ii=np.flatnonzero(meta.condition.eq(tag));metrics=row_metrics(truth[ii],prediction[ii])
   aligned.append({'model':family,'condition':tag,**{k:float(np.nanmean(v)) if np.isfinite(v).any() else None for k,v in metrics.items()}})
 pd.DataFrame(aligned).to_csv(o/'aligned_ensemble_controls.csv',index=False)
 positives=pd.read_csv(o/'guide_disjoint_positive_controls.csv');assert len(positives)==112
 assert positives.groupby('condition').target.nunique().eq(28).all()
 negatives=pd.read_csv(o/'wrong_target_controls.csv');assert len(negatives)==160 and negatives.repeat.nunique()==20
 pairs=read(o/'paired_target_contrasts.json')
 for family,item in pairs.items():
  assert item['targets']==28
  assert abs(item['hierarchical_minus_comparator_MAE']-(summary['hierarchical']['Teff']['mae']-summary[family]['Teff']['mae']))<1e-10
 transfer.update(gzip_CRC='verified by two complete preprocessing streams',biological_evaluation_completed=True,updated_epoch=time.time())
 (R/'transfer_verification.json').write_text(json.dumps(transfer,indent=2),encoding='utf-8')
 audit={'state':'passed','complete_raw_count_streams':2,'gzip_CRC_verified':True,'source_files_hash_verified':len(proto['frozen_files']),'protocol_sha256':lock['protocol_sha256'],'gene_IDs_unique':True,'all_112_gold_rows_rebuilt_from_guide_counts':True,'maximum_gold_rebuild_error':gold_error,'maximum_control_rebuild_error':control_error,'output_genes':int(mask.sum()),'missing_input_genes_source_imputed_and_output_masked':26,'all_conditions_balanced_targets':28,'all_seed_condition_rows':len(perseed),'all_target_model_seed_rows':len(frame),'wrong_target_rows':len(negatives),'positive_control_rows':len(positives),'positive_control_retrieval_warning':'singleton pair calculations have trivial retrieval=1; not a valid panel retrieval estimate and not reported as such','external_response_training_or_selection':False,'not_independent_donor_or_laboratory_validation':True,'updated_epoch':time.time()}
 (o/'postrun_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
