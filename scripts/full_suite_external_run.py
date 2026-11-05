"""Execute the prespecified, explicitly confounded E3 source comparison."""
from full_suite_core import *
from full_suite_run import run_task,freeze_plan
from full_suite_external_prepare import aggregate
DEST=OUT/'external_data'
def main():
    eproto=json.loads((OUT/'E3_external_protocol.json').read_text(encoding='utf-8'));protocol=freeze_plan()
    for fname,name in [('ReplogleWeissman2022_K562_essential.h5ad','K562'),('ReplogleWeissman2022_rpe1.h5ad','RPE1')]:aggregate(DEST/fname,name)
    ext={k:np.load(DEST/(k+'_pseudobulk.npz'),allow_pickle=False) for k in ('K562','RPE1')}
    donor=np.load(PREP/'donor_effects.npy',mmap_mode='r');dm=pd.read_csv(PREP/'donor_rows.csv');dg=pd.read_csv(PREP/'donor_genes.csv').gene_id.astype(str).to_numpy()
    complete=dm.groupby('target_contrast').size();targets=set(complete[complete==12].index)
    for k,a in ext.items():targets &= set(a['target_ids'][a['n_cells']>=5])-{'NTC'}
    targets=sorted(targets);genes=set(dg)
    for a in ext.values():genes &= set(a['gene_ids'])
    genes=np.array(sorted(genes));gmap={g:i for i,g in enumerate(dg)};gp=np.array([gmap[g] for g in genes])
    if len(targets)<100 or len(genes)<1000:raise ValueError('Insufficient matched target/gene coverage for E3')
    subset=dm[dm.target_contrast.isin(targets)].copy().reset_index(drop=True)
    primary=np.empty((len(subset),len(genes)),np.float32)
    for start in range(0,len(subset),128):primary[start:start+128]=np.asarray(donor[subset.iloc[start:start+128].storage_row.to_numpy()])[:,gp]
    ctrl=np.load(PREP/'controls.npz',allow_pickle=False);cg={g:i for i,g in enumerate(ctrl['gene_ids'])};cp=np.array([cg[g] for g in genes])
    controls={(d,c):ctrl['control_log2cpm'][i*3+j,cp] for i,d in enumerate(ctrl['donors']) for j,c in enumerate(ctrl['contexts'])}
    aux_values=[];aux_records=[]
    for name,a in ext.items():
        gm={g:i for i,g in enumerate(a['gene_ids'])};pp=np.array([gm[g] for g in genes]);tm={t:i for i,t in enumerate(a['target_ids'])};ntc=tm['NTC']
        control=np.log2(1+a['counts'][ntc,pp]/a['library_sizes'][ntc]*1e6).astype(np.float32);controls[(name+'_cellline',name)]=control
        for t in targets:
            j=tm[t];effect=np.log2(1+a['counts'][j,pp]/a['library_sizes'][j]*1e6)-control
            aux_values.append(effect.astype(np.float32));aux_records.append({'index':t+'|'+name,'target_contrast':t,'culture_condition':name,'donor_id':name+'_cellline','partition':'train','n_cells':int(a['n_cells'][j]),'source_kind':'cell_line_not_human_donor'})
    aux=np.stack(aux_values)
    targeted_aux=set(eproto['targeted_auxiliary_contexts']);key=subset.donor_id+'|'+subset.culture_condition
    write_json(DEST/'E3_coverage.json',{'common_targets':len(targets),'common_genes':len(genes),'primary_rows':len(subset),'auxiliary_rows_each_regime':2*len(targets),'source_context_matching':'two auxiliary contexts in either regime','unmatched_confounders':eproto['unmatched_confounders'],'protocol_sha256':sha(OUT/'E3_external_protocol.json')})
    for regime in ('targeted','non_primary_T'):
        m=subset.copy();values=primary
        if regime=='non_primary_T':
            keep=~((m.partition=='train')&key.isin(targeted_aux));m=m[keep].copy();values=primary[keep.to_numpy()]
            m=pd.concat([m,pd.DataFrame(aux_records)],ignore_index=True);values=np.concatenate([values,aux],axis=0)
        m['storage_row']=np.arange(len(m));m['selection_or_calibration']=['calibration' if hashnum('calibration|'+str(t))%2 else 'selection' for t in m.target_contrast]
        m.attrs['control_override']=controls;m.attrs['hashed_context_encoding']=True
        source_counts=m[m.partition=='train'].groupby(['donor_id','culture_condition']).size().to_dict()
        write_json(DEST/(regime+'_budget.json'),{'training_rows':int((m.partition=='train').sum()),'training_target_ids':len(targets),'training_context_rows':{str(k):int(v) for k,v in source_counts.items()},'target_cell_count_median':float(m[m.partition=='train'].n_cells.median()),'new_test_donor':'CE0008678','only_matching_observations_and_computation_not_cells_or_donor_kind':True})
        run_task('E3_external_'+regime,values,m,genes,['ridge','concat','fv'],protocol,'P1')
    # Entire external response study is held out for this independent model fit.
    m=pd.concat([subset.copy(),pd.DataFrame(aux_records)],ignore_index=True)
    m.loc[(m.donor_id=='CE0008678'),'partition']='excluded'
    m.loc[m.donor_id.isin(['K562_cellline','RPE1_cellline']),'partition']='test'
    values=np.concatenate([primary,aux],axis=0);m['storage_row']=np.arange(len(m))
    m['selection_or_calibration']=['calibration' if hashnum('calibration|'+str(t))%2 else 'selection' for t in m.target_contrast]
    m.attrs['control_override']=controls;m.attrs['hashed_context_encoding']=True
    np.save(DEST/'study_holdout_effects.npy',values)
    np.savez_compressed(DEST/'study_controls.npz',keys=np.array([d+'|'+c for d,c in controls]),values=np.stack(list(controls.values())),gene_ids=genes)
    run_task('E1_external_study_P1',values,m,genes,['zero','training_mean','ridge','concat','state_official','fv'],protocol,'P1')
    write_json(DEST/'E3_status.json',{'state':'complete','scope':'exploratory; unmatched cell counts, biological unit, guide, laboratory and retention confounders remain','updated_epoch':time.time()})
if __name__=='__main__':
    lock=OUT/'external_training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
