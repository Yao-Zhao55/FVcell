"""Deterministic information-isolated split rules; no response-value inputs."""
import hashlib,math
SEED='fvcell-v2-20261025-protocol-1'
CONDITIONS=('Rest','Stim8hr','Stim48hr')
def number(namespace,key):
    return int(hashlib.sha256((SEED+'|'+namespace+'|'+key).encode()).hexdigest()[:16],16)
def combo_split(target,condition):
    group=number('pair-group',target)%100
    held=CONDITIONS[number('held-context',target)%len(CONDITIONS)]
    return ('test' if group<20 else 'validation') if group<40 and condition==held else 'train'
def target_split(target):
    value=number('unseen-target',target)%100
    return 'test' if value<20 else 'validation' if value<40 else 'train'
def donor_split(donor):
    # Generic policy only: actual frozen donor assignment is explicit in output.
    return number('donor',donor)
def observed(value):
    return isinstance(value,(int,float)) and math.isfinite(value)
def control_key(row):
    return tuple(row[k] for k in ['study_id','donor_id','context_id','assay','readout_time'])
def validate_effect_rows(rows):
    seen={}
    for row in rows:
        key=tuple(row[k] for k in ['study_id','donor_id','context_id','target_id','response_gene_id'])
        if key in seen:raise ValueError('Duplicate effect key; resolve replicate/provenance explicitly')
        seen[key]=row
        if row['observed']!=observed(row['effect']):raise ValueError('Mask inconsistent with finite measurement')
        if row['observed'] and row['source_kind']!='measured_effect':raise ValueError('Predicted/fallback effect cannot be gold label')
    return True
