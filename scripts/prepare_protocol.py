"""Build measured coverage, source eligibility and frozen split manifests."""
from pathlib import Path
import pandas as pd,json,gzip,re,hashlib,xml.etree.ElementTree as ET
from collections import defaultdict
from split_rules import SEED,CONDITIONS,combo_split,target_split,donor_split
BASE=Path(__file__).resolve().parents[1]
RAW=BASE/'sources';OUT=BASE/'prepared';OUT.mkdir(exist_ok=True)
def write_json(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(frame,name):frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')

tables=[]
for path in (RAW/'selected_tables').glob('*.xlsx'):
    frame=pd.read_excel(path)
    summary={'source_file':path.name,'rows':len(frame),'columns':frame.columns.tolist()}
    if 'padj' in frame:
        summary.update(finite_padj=int(frame.padj.notna().sum()),padj_le_01=int((frame.padj<=.1).sum()),padj_gt_01=int((frame.padj>.1).sum()))
    if 'S9_' in path.name:
        summary.update(targets=int(frame.KO.nunique()),response_genes=int(frame.gene_name.nunique()),source_role='significance_selected_mechanistic_support_only',eligible_complete_response_gold=False)
        frame=frame.assign(study_id='GSE278572',context_id=frame.cell_type.astype(str)+'_'+frame.stimulation.astype(str),effect_unit='DESeq2_log2FC',measurement_status='observed_significance_selected',source_kind='measured_effect')
        csv(frame,'GSE278572_significant_support.csv')
        cov=frame.groupby(['context_id','KO']).agg(observed_significant_genes=('gene_name','nunique')).reset_index()
        cov['donor_id']='pooled_or_not_recoverable_from_S9';cov['complete_assayed_gene_set']=False
        csv(cov,'GSE278572_support_coverage.csv')
    else:
        summary.update(source_role='transcriptome_derived_activation_summary_not_independent_functional_validation',contexts=sorted(frame.HTO_maxID.unique().tolist()),targets=int(frame.sg_target.nunique()))
        csv(frame,'GSE278572_activation_summary.csv')
    tables.append(summary)
write_json('source_table_audit.json',tables)

samples=pd.read_csv(RAW/'GWT_sample_metadata.suppl_table.csv')
safe=samples[['cell_sample_id','10xrun_id','donor_id','culture_condition','library_id','library_prep_kit','sequencing_platform']].copy()
safe['study_id']='GSE314342';safe['tissue']='peripheral_blood_derived_ex_vivo';safe['cell_type']='primary_CD4_T';safe['intervention']='CRISPRi';safe['modality']='scRNA-seq';safe['organ_intervention_validation']=False
csv(safe,'GSE314342_donor_context_coverage.csv')
donors=sorted(safe.donor_id.unique(),key=donor_split)
assignment={d:('test' if i==0 else 'validation' if i==1 else 'train') for i,d in enumerate(donors)}
donor_rows=safe.copy();donor_rows['partition']=donor_rows.donor_id.map(assignment)
donor_rows['label_readiness']='prospective_only_donor_resolved_counts_or_independent_effects_required'
csv(donor_rows,'donor_split_prospective.csv')
write_json('donor_split_lock.json',{'seed':SEED,'assignment':assignment,'evaluation_status':'not_activated_no_donor_resolved_effect_labels','pooled_DE_stats_cannot_be_assigned_to_one_donor':True,'all_tissues_and_technical_replicates_of_donor_must_share_partition':True})

# Parse GEO sample records. These are library declarations, not per-cell tissue assignments.
soft=gzip.open(RAW/'GSE299043_family.soft.gz','rt',encoding='utf-8').read()
atlas=[]
for block in soft.split('^SAMPLE = ')[1:]:
    gsm=block.splitlines()[0].strip()
    fields=defaultdict(list)
    for line in block.splitlines():
        if line.startswith('!Sample_') and ' = ' in line:
            k,v=line.split(' = ',1);fields[k].append(v)
    chars={}
    for text in fields.get('!Sample_characteristics_ch1',[]):
        if ': ' in text:
            k,v=text.split(': ',1);chars[k.lower()]=v
    title=fields.get('!Sample_title',[''])[0]
    tissue=chars.get('tissue',fields.get('!Sample_source_name_ch1',['unknown'])[0])
    modality='ADT' if '_ADT_' in title else 'RNA' if '_RNA_' in title else 'TCR' if 'TCR' in title else 'BCR' if 'BCR' in title else 'unspecified_library'
    atlas.append(dict(study_id='GSE299043',sample_id=gsm,library_title=title,donor_id=chars.get('donor id','unknown'),declared_tissues=tissue,pooled_tissue_library=tissue.startswith('pooled:'),modality=modality,cell_type=chars.get('enriched cell_type','unknown'),intervention='observational_no_experimental_perturbation',readout_time='cross_sectional',coverage_level='library_declared_not_cell_demultiplexed',organ_perturbation_gold=False))
atlas=pd.DataFrame(atlas)
csv(atlas,'GSE299043_library_coverage.csv')
organ_rows=[]
for r in atlas.itertuples():
    for tissue in r.declared_tissues.removeprefix('pooled:').split(','):
        organ_rows.append(dict(donor_id=r.donor_id,declared_tissue=tissue.strip(),sample_id=r.sample_id,modality=r.modality,coverage_level=r.coverage_level,actual_cell_count='not_available'))
csv(pd.DataFrame(organ_rows).drop_duplicates(),'GSE299043_declared_donor_tissue_modality.csv')

# Independent measured validation remains a held-out evidence layer, not a training feature.
function=pd.read_csv(RAW/'GWT_IL10_IL21_arrayed_validation_combined.csv')
csv(function,'external_IL10_IL21_measured_flow_holdout.csv')
write_json('functional_holdout_lock.json',{'source':'GSE314342 author supplementary IL10_IL21_arrayed_validation_combined.csv','rows':len(function),'donors':sorted(function.Donor.unique()),'endpoint_columns':['IL10_perc','IL21_perc'],'control_column':'Perturbation','control_value':'NTC','role':'independent_assay_and_donors_within_primary_study_not_independent_study','source_perturbations':sorted(function.Perturbation.unique()),'labels_not_used_in_training_or_model_selection':True,'prospective_test_only_no_model_results_computed':True,'not_identical_endpoint_to_transcriptome_delta':True})

# Full small row-index must be complete and match the pinned author repository.
index_file=RAW/'GWT_DE_stats.suppl_table.csv'
index_status={'available':False}
manifest=json.loads((BASE/'cloud_download_manifest.json').read_text(encoding='utf-8'))
item=next(x for x in manifest['files'] if x['name']==index_file.name)
if index_file.exists() and index_file.stat().st_size==item['bytes']:
    data=index_file.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob!=item['git_blob_sha1']:raise RuntimeError('Condition index checksum does not match pinned author version')
    frame=pd.read_csv(index_file)
    if frame['index'].duplicated().any():raise RuntimeError('Duplicate DE observation ID')
    eligible=frame[frame.culture_condition.isin(CONDITIONS)].copy()
    complete=eligible.groupby('target_contrast').culture_condition.agg(lambda x:set(x)==set(CONDITIONS))
    eligible=eligible[eligible.target_contrast.isin(complete[complete].index)].copy()
    # No gene/target selection by downstream response, significance or guide/donor correlation.
    columns=['index','target_contrast','target_contrast_gene_name','culture_condition']
    split=eligible[columns].sort_values('index').copy()
    split['combo_partition']=[combo_split(t,c) for t,c in zip(split.target_contrast,split.culture_condition)]
    split['global_target_partition']=split.target_contrast.map(target_split)
    split['study_id']='GSE314342';split['effect_labels_ready']=False
    csv(split,'condition_and_global_target_splits.csv')
    summary=eligible.groupby('culture_condition').agg(targets=('target_contrast','nunique'),conditions=('index','size')).reset_index()
    csv(summary,'GSE314342_response_condition_coverage.csv')
    excluded=frame[~frame['index'].isin(eligible['index'])][columns].copy();excluded['reason']='not_present_in_all_three_contexts_or_context_not_in_protocol'
    csv(excluded,'excluded_condition_rows.csv')
    index_status={'available':True,'source_rows':len(frame),'eligible_rows':len(split),'eligible_targets':int(split.target_contrast.nunique()),'excluded_rows':len(excluded),'combo_counts':split.combo_partition.value_counts().to_dict(),'global_target_counts':split.global_target_partition.value_counts().to_dict(),'checksum':blob,'full_effect_labels_local':False}
write_json('condition_index_status.json',index_status)

schema={'primary_unit':'published_DESeq2_log2_fold_change','primary_label_kind':'estimated_terminal_response_not_raw_counts_or_single_cell_paired_difference','required_fields':['study_id','sample_id','donor_id','context_id','assay','intervention_type','target_id','response_gene_id','effect','effect_unit','standard_error','observed','source_kind','control_definition','readout_time','source_version','source_hash'],
 'source_kind_values':['measured_effect','raw_counts','mechanistic_prior','teacher_prediction','fallback_estimate'],
 'mask_rule':'finite measured estimates, including zero and non-significant estimates; unmeasured and missing stay missing',
 'significance_threshold_is_not_observation_mask':True,
 'control_matching_fields':['study_id','donor_id','context_id','assay','readout_time'],
 'unresolved_controls_block_primary_labels':True,
 'study_aggregated_effect_donor_id':'pooled_not_donor_resolved',
 'baseMean_is_not_control_expression':True,
 'different_effect_units_must_not_be_pooled':True}
write_json('data_schema_v1.json',schema)
protocol={'protocol_id':'FVcell-v2-day25_26-v1','freeze_date':'2026-10-25','seed':SEED,'scope':'ex_vivo_primary_CD4_T_cell_context_response_transfer','title':'FVcell: Context-Conditioned Perturbation Response Transfer in Primary T Cells','primary_study':'GSE314342','primary_action':'CRISPRi_not_complete_gene_knockout','contexts':list(CONDITIONS),'primary_supervised_unit':'published_DESeq2_log2FC_all_assayed_genes','gene_universe_rule':'all measured genes in pinned primary DE artifact var; no target-response-driven HVG/top-DEG selection','gene_id_manifest_status':'pending_var_extraction_before_training','primary_metric':'unweighted_signed_log2FC_MAE_macro_by_context_then_target','secondary_metrics':['signed_delta_Pearson_with_constant_vector_NA','direction_accuracy_deadzone_abs_truth_0.1_log2FC','average_precision_DEG_FDR_0.1_with_prevalence'], 'primary_partition':'unseen_context_target_combination','primary_partition_permissions':'P2 permits other training contexts of the same target; no held-out condition response','P1_requires_independent_non_targeting_baseline_not_baseMean':True,
 'global_unseen_target':'secondary_all_sources_remove_held_target_effects_and_teacher_predictions','donor_test_status':'prospective_only_until_donor_resolved_labels_exist','external_study_status':'not_activated_significance_only_S9_cannot_supply_complete_gold','function_test':'locked measured IL10_IL21 percentages, independent assay not independent study; no results claimed','excluded_predictors':['n_up_genes','n_down_genes','n_total_de_genes','n_downstream','ontarget_effect_size','guide_correlation_all','donor_correlation_all_mean','test_condition_baseMean','test_significant_gene_lists'],
 'pretrained_existing_npz_role':'read_only_provenance_audit_only_not_teacher_or_gold_due_pooled_test_exposure_and_selection','gold_readiness_gate':['all_gene_effect_labels_obtained_and_validated','control_definition_confirmed_from_author_analysis','effect_units_verified','training_only_fit_of_transformations','variable_gene_manifest_frozen_before_model_training'],
 'baselines':['zero_response','training_global_mean','same_target_other_context_mean_P2','ridge_or_low_rank_linear','capacity_matched_context_MLP','task_matched_reproducible_strong_baseline'],
 'organ_and_dynamic_claims':'not_in_primary_claims_without_matched_organ_perturbation_or_longitudinal_gold'}
protocol['author_control_audit']={'control_semantics':'non_targeting_guide','source_code_default_literal':'NO-TARGET','source_validation_table_literal':'NTC','pseudobulk_literal_status':'must_resolve_from_downloaded_obs_before_labels_activated','contrast':'target_minus_control_within_culture_condition','author_design':'~ log10_n_cells + donor_id + target','donor_covariate_is_not_donor_resolved_gold':True,'author_upstream_n_hvgs':10000,'upstream_feature_selection_limitation':'published artifact includes author feature selection; no claim of independently training-only selected gene universe'}
write_json('protocol_v1_frozen.json',protocol)
write_json('day25_26_summary.json',{'small_tables':tables,'primary_samples':len(safe),'primary_donors':len(donors),'atlas_libraries':len(atlas),'atlas_declared_donors':int(atlas.donor_id.nunique()),'atlas_pooled_tissue_libraries':int(atlas.pooled_tissue_library.sum()),'functional_measurement_rows':len(function),'condition_index':index_status,'storage_decision':'local_two_required_processed_files_61352897247_bytes_under_100GB_cloud_deferred','complete_effect_matrix_status':'download_status_json_is_authoritative_not_yet_activated_as_gold','pseudobulk_status':'download_status_json_is_authoritative_pending_control_and_donor_checks'})
print(json.dumps({'index':index_status,'atlas_libraries':len(atlas),'atlas_donors':int(atlas.donor_id.nunique()),'source_tables':tables},ensure_ascii=False),flush=True)
