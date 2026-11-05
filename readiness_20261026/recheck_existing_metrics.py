"""Reaggregate existing held-donor scores. No inference, training or new gold."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json
import numpy as np,pandas as pd
from complete_priorities import ROOT,BASE,sha,equal_donor_mae

def main():
    # Preserve the frozen file. Record a visible correction from source-code audit.
    amendment={
     'version':'FVcell-supplement-v1-amendment-01',
     'base_protocol_sha256':sha(ROOT/'evaluation_protocol_frozen.json'),
     'reason':'Source-code audit found that the existing DE AP uses finite published adj_p_value <= 0.1 without an effect-size cutoff. The newly proposed 0.05/0.5 definition would change the existing endpoint. Preserve the original definition rather than silently introducing a new ranking.',
     'override':'supplemental_metrics[DE_AP]',
     'rule':'finite published adj_p_value <= 0.1; rank absolute predicted log2FC; no effect-size filtering; report prevalence and no-positive conditions; no donor DE calls invented',
     'evidence_file':'scripts/full_suite_run.py',
     'evidence_sha256':sha(BASE/'scripts/full_suite_run.py'),
     'affects_primary_endpoint':False,
     'new_definition_005_05_not_executed':True,
     'reaggregation':'four fixed donor folds x four final families x three fixed seeds; MAE/Pearson/direction only from retained test_rows.csv; no weight update or new test selection',
     'interpretation':'exploratory recheck of already-observed outputs, not independent confirmation'
    }
    p=ROOT/'evaluation_protocol_amendment_01.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==amendment
    else:p.write_text(json.dumps(amendment,ensure_ascii=False,indent=2),encoding='utf-8')
    ap=ROOT/'AMENDMENT_LOCK.json'
    value={'base_protocol_sha256':amendment['base_protocol_sha256'],'amendment_sha256':sha(p)}
    if ap.exists():assert json.loads(ap.read_text(encoding='utf-8'))==value
    else:ap.write_text(json.dumps(value,indent=2),encoding='utf-8')
    records=[];inputs=[]
    for fold in range(4):
        for family in ['hierarchical','fixed_gaussian','unshrunk_offsets','fv_continued']:
            for seed in [11,29,47]:
                source=BASE/'training_completion'/'runs'/f'hierarchical_donor_fold{fold}'/family/str(seed)/'test_rows.csv'
                f=pd.read_csv(source)
                assert len(f)>0 and f.donor.nunique()==1 and not f['index'].duplicated().any()
                def macro(col):
                    return float(f.groupby(['donor','context','target'])[col].mean().groupby(['donor','context']).mean().groupby('donor').mean().mean())
                records.append({'fold':fold,'held_donor':str(f.donor.iloc[0]),'family':family,'seed':seed,'rows':len(f),'targets':f.target.nunique(),'mae':equal_donor_mae(f),'signed_Pearson':macro('pearson'),'direction_accuracy':macro('direction_accuracy'),'undefined_Pearson_rows':int(f.pearson.isna().sum()),'undefined_direction_rows':int(f.direction_accuracy.isna().sum())})
                inputs.append({'path':str(source.relative_to(BASE)),'sha256':sha(source)})
    tab=pd.DataFrame(records)
    tab.to_csv(ROOT/'rechecked_existing_results.csv',index=False,encoding='utf-8-sig')
    grouped=tab.groupby(['family','held_donor'])[['mae','signed_Pearson','direction_accuracy']].mean()
    grouped.reset_index().to_csv(ROOT/'rechecked_donor_summary.csv',index=False,encoding='utf-8-sig')
    summary={'existing_result_files_rechecked':len(inputs),'independent_donors':tab.held_donor.nunique(),'new_trained_models':0,'new_biological_validation':False,'families_equal_donor_mean':grouped.groupby('family').mean().to_dict('index'),'macro_metric_change':'Equal donor/context/target aggregation fixed; source files and original reports unchanged. These are reaggregations of existing predictions, not a fresh test.','inputs':inputs}
    # MAE must reproduce the final stored summary, not merely look plausible.
    old=pd.read_csv(BASE/'training_completion/analysis/all_results.csv')
    differences=[]
    for (fold,fam),g in tab.groupby(['fold','family']):
        stored=old[(old.task==f'hierarchical_donor_fold{fold}')&(old.family==fam)].mae.iloc[0]
        differences.append(abs(float(g.mae.mean())-float(stored)))
    summary['max_MAE_difference_from_existing_summary']=max(differences)
    assert max(differences)<1e-10,differences
    (ROOT/'metrics_recheck_status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='inputs'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
