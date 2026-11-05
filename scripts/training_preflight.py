"""Read-only readiness check; never train from an incomplete/preallocated file."""
from pathlib import Path
import json,hashlib,datetime
BASE=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=load(BASE/'large_asset_manifest.json')
freeze=load(BASE/'prepared/FREEZE_LOCK.json')
checks={name:sha(BASE/'prepared'/name)==value for name,value in freeze['sha256'].items()}
assets={}
for key in ('de_stats','pseudobulk'):
    meta=manifest[key];p=BASE/'large_data'/Path(meta['key']).name
    receipt=p.with_name(p.name+'.verified.json')
    complete=False
    if p.exists() and receipt.exists():
        r=load(receipt)
        complete=p.stat().st_size==meta['bytes'] and r.get('etag')==meta['etag'] and r.get('sha256')==sha(p)
    assets[key]={'verified_complete':complete,'required_bytes':meta['bytes']}
blockers=[]
if not all(checks.values()):blockers.append('Frozen protocol/split hash mismatch')
for name,info in assets.items():
    if not info['verified_complete']:blockers.append(name+': full verified data not local')
validation_path=BASE/'training_first/input_validation.json'
if not validation_path.exists():
    blockers.append('Actual gene IDs, effect layers, controls and finite observation mask have not been validated')
else:
    validation=load(validation_path)
    if validation.get('status')!='validated' or validation.get('freeze_hash')!=sha(BASE/'prepared/FREEZE_LOCK.json') or validation.get('gene_id_sha256')!=sha(BASE/'training_first/genes.csv'):
        blockers.append('Input validation is stale or gene/split hash does not match')
train_status_path=BASE/'training_first/status.json'
train_status=load(train_status_path) if train_status_path.exists() else None
result={'checked_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'formal_training_ready':not blockers,'training_started':bool(train_status and train_status.get('state') in ['training_first_additive_model','complete']),'training_status':train_status,'frozen_hash_checks':checks,'assets':assets,'blockers':blockers,'download_status':load(BASE/'download_status.json'),'next_step':'local supervisor launches validation and first experiment after verified download completion; app heartbeat diagnoses failures'}
(BASE/'prepared/training_preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
