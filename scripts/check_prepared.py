"""Check the actual split, freeze protocol hashes and verify protected inputs."""
from pathlib import Path
import hashlib,json
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]
OUT=BASE/'prepared'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
s=pd.read_csv(OUT/'condition_and_global_target_splits.csv')
assert len(s)==33258 and s.target_contrast.nunique()==11086
assert not s['index'].duplicated().any()
assert s.groupby('target_contrast').culture_condition.nunique().eq(3).all()
assert s.groupby('target_contrast').global_target_partition.nunique().eq(1).all()
assert s[s.combo_partition=='train'].groupby('target_contrast').size().ge(2).all()
assert not s.effect_labels_ready.any()
assert not set(['n_up_genes','n_down_genes','guide_correlation_all']) & set(s.columns)
d=pd.read_csv(OUT/'donor_split_prospective.csv')
assert d.groupby('donor_id').partition.nunique().eq(1).all()
assert d.groupby('partition').donor_id.nunique().to_dict()=={'test':1,'train':2,'validation':1}
checks=[]
for item in json.loads((BASE/'protected_files_before.json').read_text(encoding='utf-8')):
    p=ROOT/item['path']
    checks.append({'path':item['path'],'unchanged':p.exists() and sha(p)==item['sha256']})
assert all(x['unchanged'] for x in checks),'Protected input changed'
lockfiles=['protocol_v1_frozen.json','data_schema_v1.json','condition_and_global_target_splits.csv','donor_split_lock.json','functional_holdout_lock.json']
lock={'protocol_version':'FVcell-v2-day25_26-v1','freeze_date':'2026-10-25','status':'split_and_protocol_frozen_effect_labels_not_yet_activated','sha256':{x:sha(OUT/x) for x in lockfiles}}
(OUT/'FREEZE_LOCK.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2),encoding='utf-8')
result={'actual_data_split_checks':'passed','protected_files_checked':len(checks),'protected_files_unchanged':all(x['unchanged'] for x in checks),'eligible_rows':len(s),'targets':int(s.target_contrast.nunique()),'no_model_training_or_biological_accuracy_claim':True}
(OUT/'verification_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
