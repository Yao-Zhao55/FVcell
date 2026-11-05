"""Remove only this run's reproducible redundant prediction caches.

Keep all raw data, fitted models, per-target metrics, summaries and provenance.
Keep the main/concat combination arrays needed for uncertainty, and E5 arrays.
Never recurse outside the newly generated full_experiments/runs tree.
"""
from pathlib import Path
import json,time
ROOT=(Path(__file__).resolve().parents[1]/'full_experiments/runs').resolve()
journal=ROOT.parent/'prediction_cache_retention.json'
old=json.loads(journal.read_text(encoding='utf-8')) if journal.exists() else {'removed':[]}
freed=0
for task in ROOT.iterdir():
    if not task.is_dir():continue
    for family in task.iterdir():
        if not family.is_dir() or not (family/'summary.json').exists():continue
        keep=(task.name=='E1_P2_combo' and family.name in ('fv','concat')) or task.name.startswith('E5_')
        if keep:continue
        for seed in (11,29,47):
            folder=family/str(seed)
            if not (folder/'result.json').exists() or not (folder/'model.pt').exists():continue
            for name in ('test_predictions.npy','validation_predictions.npy'):
                p=(folder/name).resolve()
                if not p.is_relative_to(ROOT):raise ValueError('Unexpected cache path')
                if p.is_file():
                    size=p.stat().st_size;p.unlink();freed+=size;old['removed'].append({'path':str(p.relative_to(ROOT)),'bytes':size,'reason':'reproducible prediction cache; checkpoint and per-target metrics retained','epoch':time.time()})
old['total_freed_bytes']=sum(x['bytes'] for x in old['removed'])
journal.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'freed_this_run_GB':freed/1e9,'total_freed_GB':old['total_freed_bytes']/1e9}))
