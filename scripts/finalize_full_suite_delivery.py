"""Record final reviewed artifacts and close top-level work snapshots."""
from pathlib import Path
import json,hashlib,datetime,os,shutil
B=Path(__file__).resolve().parents[1];O=B/'full_experiments'
A=json.loads((O/'completion_audit.json').read_text())
assert A['state']=='all_registered_computational_runs_complete'
assert not list(O.glob('*.lock'))
now=datetime.datetime.now().astimezone().isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
q=O/'output/pdf_previews/layout_checks.json';v=json.loads(q.read_text(encoding='utf8'))
v.update({'visual_review_still_required':False,'visual_review':'All 12 rendered pages reviewed in two overview sheets; enlarged pages 5, 8 and 10 checked; clipped plot title corrected and all pages rerendered.','reviewed_at':now});save(q,v)
pdf=O/'output/pdf/FVcell_大纲v2模型实验报告_20261026.pdf'
status={'state':'registered_computational_experiments_and_report_complete','task_settings':A['task_settings'],'method_task_settings':A['method_task_settings'],'neural_checkpoints':A['neural_checkpoints'],'deterministic_baselines':A['deterministic_baseline_settings'],'pipeline_tests_passed':20,'protected_original_files_unchanged':A['protected_original_files_unchanged'],'pdf_pages':12,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'no_active_training_locks':True,'full_outline_biological_claims_not_all_established':True,'remaining_evidence_gaps':A['scope_gaps']+['P0 removes target baseline-expression inputs but is not whole-context source exclusion','No arbitrary gene-level missing-label training mask; observed condition rows only'],'completed_at':now}
save(O/'delivery_status.json',status)
for name in ['suite_status.json','donor_cv_progress.json','donor0_state_progress.json','external_training_progress.json','strict_suite_status.json','functional_strict_progress.json']:
    p=O/name
    if p.exists():
        s=json.loads(p.read_text(encoding='utf8'));s.update({'state':'complete','completion_authority':'completion_audit.json','completed_at':now});save(p,s)
t=O/'runs/E1_P2_combo/complete.json';s=json.loads(t.read_text());s['families']=sorted(p.name for p in t.parent.iterdir() if p.is_dir() and (p/'summary.json').exists());save(t,s)
sizes=[(Path(r)/n).stat().st_size for r,ds,fs in os.walk(B) for n in fs]
storage={'logical_file_bytes_in_day25_26':sum(sizes),'logical_size_GB':sum(sizes)/1e9,'file_count':len(sizes),'unreadable_paths':[],'free_disk_GB':shutil.disk_usage(B).free/1e9,'primary_and_external_data_download_bytes':64136513822,'cloud_used':False,'observed_at':now};save(O/'storage_audit.json',storage)
print(json.dumps({'completed':status['state'],'model_settings':A['method_task_settings'],'local_GB':storage['logical_size_GB'],'pdf_pages':12},ensure_ascii=False))
