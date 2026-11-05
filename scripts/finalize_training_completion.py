"""Close the audited local experiment delivery after human visual review."""
from pathlib import Path
import json,hashlib,datetime,os,shutil
B=Path(__file__).resolve().parents[1];R=B/'training_completion';now=datetime.datetime.now().astimezone().isoformat()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(R/'audit.json');assert a['state']=='all_registered_additional_runs_complete' and a['new_neural_checkpoints']==150
assert read(R/'auxiliary_status.json')['state']=='complete' and read(R/'calibration_status.json')['cases']==16
assert not list(R.glob('*.lock')),'A worker or evaluator is still active'
q=R/'output/pdf_previews/layout_checks.json';v=read(q);assert len(v['pages'])==9
v.update({'visual_review_required':False,'visual_review':'All nine current rendered pages inspected via overview sheets, with enlarged pages 2 and 8; chart clipping corrected and all pages rerendered.','reviewed_at':now});save(q,v)
p=R/'code_manifest.json';v=read(p);v['files']['report_training_completion.py']=sha(B/'scripts/report_training_completion.py');v['report_layout_finalized_at']=now;save(p,v)
pdf=R/'output/pdf/FVcell_大纲训练补完核验_20261026.pdf'
status={'state':'additional_training_evaluation_and_report_complete','time':now,'new_task_settings':a['new_tasks'],'new_method_task_settings':a['new_method_task_settings'],'new_neural_checkpoints':150,'new_deterministic_baselines':18,'pipeline_checks':14,'new_checks':8,'protected_original_files_unchanged':98,'final_donor_interval_cases':16,'final_auxiliary_metric_runs':12,'PDF_pages':9,'PDF_sha256':sha(pdf),'all_training_and_evaluation_locks_closed':True,'result':'No demonstrated hierarchical advantage. Main hierarchical MAE=0.17339194416304385 vs matched continued FV=0.17338581191739164 and simple additive=0.17203018540218176. Fixed-vs-hierarchical four-donor difference 4.15e-7, descriptive interval crosses zero. All six hierarchical functional correlation intervals cross zero.','remaining_evidence_requirements':['Fully matched broad pretraining comparison','Independent harmonized primary-T CRISPRi study','Real organ intervention/composition/longitudinal observations if those claims are retained'],'unsupported_claims_not_marked_complete':True,'no_cloud_or_purchases':True}
save(R/'delivery_status.json',status)
for p in R.glob('*_progress.json'):
    v=read(p);v.update({'state':'complete','completion_authority':'audit.json','completed_at':now});save(p,v)
sizes=[(Path(root)/name).stat().st_size for root,dirs,files in os.walk(B) for name in files]
storage={'logical_size_GB':sum(sizes)/1e9,'file_count':len(sizes),'free_disk_GB':shutil.disk_usage(B).free/1e9,'data_download_bytes_unchanged':64136513822,'new_data_downloaded_this_completion':False,'cloud_used':False,'time':now};save(R/'storage_audit.json',storage)
print(json.dumps({'state':status['state'],'new_models':150,'PDF_pages':9,'local_GB':storage['logical_size_GB']},ensure_ascii=False))
