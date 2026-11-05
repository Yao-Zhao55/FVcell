from pathlib import Path
import hashlib,json,datetime
R=Path(__file__).resolve().parent;B=R.parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while v:=f.read(8*1024*1024):h.update(v)
 return h.hexdigest()
P={
 'version':'external-GSE278572-v1-20261027','registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'dataset':'GSE278572; Arce et al.; separate experiment from GWT; overlapping laboratories, not a lab-held-out test',
 'verified_intervention':'Zim3-dCas9 CRISPRi for Perturb-CITE-seq, despite knockout experiments elsewhere in the paper',
 'study_donors':'Two donors pooled before sequencing; identities unavailable, disjointness from GWT cannot be established',
 'source_training':'Existing read-only GWT P1 donor checkpoints; no new model selection or fine-tuning using this study',
 'test_primary':['Resting_Teff','Stimulated_Teff'],'test_secondary':['Resting_Treg','Stimulated_Treg'],
 'context_mapping':{'Resting_Teff':'Rest','Stimulated_Teff':'Stim48hr','Resting_Treg':'Rest','Stimulated_Treg':'Stim48hr'},
 'cell_filter':{'one_guide_from_cellranger':True,'minimum_GEX_UMI':1000,'minimum_GEX_features':500,'maximum_mito_fraction':0.2,'hashtag_top_UMI_minimum':50,'hashtag_top_to_runner_ratio_minimum':3},
 'cell_filter_basis':'Fixed count/QC criteria before reading matrix; no perturbation effect, significance, or model score filtering',
 'hashtag_mapping':'Feature IDs HTO_1..4 and author-provided feature names; not inferred from response expression',
 'target_minimum_cells_per_condition':50,'NTC_minimum_cells_per_condition':200,
 'target_universe':'All unique mapped non-control single-guide targets meeting fixed count-only criteria; targets may have been seen in GWT',
 'gene_universe':'Intersection of all source model reference genes with external Gene Expression Ensembl IDs, no DE/HVG filtering on external outcomes',
 'missing_inputs':'Unmeasured control input genes imputed by mean of SOURCE training-donor control log2CPM, separately by culture; never impute missing gold',
 'unmeasured_output_genes':'Masked out, never assigned zero truth',
 'unit':'log2(1+CPM_target)-log2(1+CPM_NTC); library size uses all external Gene Expression features, excluding antibody and guide counts',
 'aggregation':'Pool QC counts over cells and guides within target x hashtag; mixed two donors are one experimental aggregate, not two independent replicates',
 'primary_checkpoint_group':'hierarchical_donor_fold0 with three prespecified seeds; fold0 fixed before observing external gold',
 'neural_models':['hierarchical','fixed_gaussian','fv_continued','concat_original_P1'],
 'baselines':['zero','source_training_mean','source_same_target_mean','source_ridge_P1','source_shrunk_additive'],
 'secondary_checkpoint_groups':'Other three existing donor folds, descriptive sensitivity if feasible; no best-fold selection',
 'metrics':{'primary':'equal hashtag then equal target MAE over all observed shared genes','secondary':['centered Pearson delta','direction accuracy for absolute gold >0.1','truth-aligned amplitude','within-condition cosine target retrieval with average ties'], 'uncertainty':'2000 paired target-cluster bootstrap draws; descriptive across prespecified targets, not a population-donor confidence interval'},
 'controls':{'wrong_target':20,'permutation_seed':20261027,'positive':'Guide-disjoint target and NTC count aggregates on same gene universe; guides sorted then alternate split; report availability and measurement agreement'},
 'prohibited':['test-driven hyperparameters','significance-filtered gold','RNA response as teacher','select best tissue/fold/seed','claim independent laboratory','claim donor identities verified disjoint','treat guide/cell resampling as donor replication'],
 'prior_exposure':'An older significance-only small target panel from this publication was viewed in earlier work. Complete counts, full-vector gold and this study score have not been read; this is an exploratory external extension, not pristine preregistration.',
 'download_bytes':3857480145,'project_total_cap_bytes':100000000000,
 'authors_supplied':{'ordered':[{'name':'朱棣','affiliation':'复旦大学'},{'name':'赵耀','affiliation':'复旦大学'}],'corresponding':'待定','other_authors':'待定','English_spelling':'pending author confirmation'}
}
if __name__=='__main__':
 out=R/'protocol_frozen.json'
 if out.exists():raise SystemExit('Protocol exists: do not overwrite')
 files=[]
 for family in ['hierarchical','fixed_gaussian','fv_continued']:
  for seed in [11,29,47]:files.append(B/f'training_completion/runs/hierarchical_donor_fold0/{family}/{seed}/model.pt')
 for seed in [11,29,47]:files.append(B/f'full_experiments/runs/E1_donor_P1/concat/{seed}/model.pt')
 files.append(B/'full_experiments/runs/E1_donor_P1/ridge/ridge_model.npz')
 for f in ['GSE278572_features.tsv.gz','GSE278572_protospacer_calls_per_cell.csv.gz','GSE278572_barcodes.tsv.gz','gsm_example.txt']:files.append(R/'sources'/f)
 P['frozen_files']={str(p.relative_to(B)):sha(p) for p in files}
 out.write_text(json.dumps(P,ensure_ascii=False,indent=2),encoding='utf-8')
 (R/'FREEZE_LOCK.json').write_text(json.dumps({'protocol_sha256':sha(out),'frozen_utc':P['registered_utc'],'matrix_outcomes_inspected':False},indent=2),encoding='utf-8')
 print('Frozen protocol and',len(files),'source/checkpoint hashes')
