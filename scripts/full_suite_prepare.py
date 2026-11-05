"""Build full-suite inputs without changing the original project or first run."""
from pathlib import Path
import sys,json,hashlib,time,os
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'full_experiments';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'vendor_ml'))
from run_first_training import h5py,column,index,rows,sha
from atomic_json import write_json
import numpy as np,pandas as pd
PREP=OUT/'prepared';PREP.mkdir(exist_ok=True)
def status(stage,**kw):write_json(OUT/'preparation_status.json',{'stage':stage,'updated_epoch':time.time(),**kw})
def mechanism(gene_ids,edges,seed=20261026):
    names=sorted(set(p for g,p in edges));pidx={p:i for i,p in enumerate(names)}
    # Pathway identity keeps the same projection in real and swapped graphs.
    vectors=np.stack([np.random.default_rng(int(hashlib.sha256((str(seed)+'|'+p).encode()).hexdigest()[:8],16)).normal(size=32) for p in names]).astype(np.float32)/np.sqrt(32)
    gidx={g:i for i,g in enumerate(gene_ids)};out=np.zeros((len(gene_ids),32),np.float32);degree=np.zeros(len(gene_ids),np.int32)
    for g,p in edges:
        if g in gidx:out[gidx[g]]+=vectors[pidx[p]];degree[gidx[g]]+=1
    out/=np.sqrt(np.maximum(degree[:,None],1))
    # Isolated genes get a fixed, data-independent ID vector, not zero collapse.
    for i,g in enumerate(gene_ids):
        if degree[i]==0:out[i]=np.random.default_rng(int(hashlib.sha256(g.encode()).hexdigest()[:8],16)).normal(size=32)/np.sqrt(32)
    return out,degree
def main():
    manifest=json.loads((BASE/'large_asset_manifest.json').read_text(encoding='utf-8'))
    paths={k:BASE/'large_data'/Path(v['key']).name for k,v in manifest.items() if k in ('de_stats','pseudobulk')}
    frozen=json.loads((BASE/'prepared/FREEZE_LOCK.json').read_text(encoding='utf-8'))
    for name,value in frozen['sha256'].items():
        if sha(BASE/'prepared'/name)!=value:raise ValueError('Frozen input changed: '+name)
    split=pd.read_csv(BASE/'prepared/condition_and_global_target_splits.csv').set_index('index')
    with h5py.File(paths['de_stats'],'r') as f:
        ids=index(f['obs']);selected=[x for x in ids if x in split.index]
        meta=split.loc[selected].reset_index();genes=index(f['var'])
        gene_names=column(f['var']['gene_name']) if 'gene_name' in f['var'] else genes
        pd.DataFrame({'gene_id':genes,'gene_name':gene_names}).to_csv(PREP/'primary_genes.csv',index=False,encoding='utf-8-sig')
        meta['selection_or_calibration']=['calibration' if int(hashlib.sha256(('calibration|'+str(t)).encode()).hexdigest()[:8],16)%2 else 'selection' for t in meta.target_contrast]
        meta.to_csv(PREP/'primary_rows.csv',index=False,encoding='utf-8-sig')
    p=OUT/'Ensembl2Reactome.txt';partial=p.with_name(p.name+'.download')
    if not p.exists():
        if not partial.exists():raise ValueError('Reactome download missing')
        if partial.stat().st_size>200_000_000:raise ValueError('Reactome budget exceeded')
        partial.replace(p)
    edges=set()
    with p.open(encoding='utf-8') as f:
        for line in f:
            fields=line.rstrip('\n').split('\t')
            if len(fields)==6 and fields[5]=='Homo sapiens' and fields[0].startswith('ENSG'):edges.add((fields[0],fields[1]))
    if len(edges)<1000:raise ValueError('Invalid Reactome human mapping')
    all_targets=sorted(meta.target_contrast.unique());embedding,degree=mechanism(all_targets,edges)
    np.savez_compressed(PREP/'mechanism.npz',target_ids=np.array(all_targets),embedding=embedding,degree=degree)
    # Bipartite double-edge swaps preserve gene and pathway degrees exactly.
    targetset=set(all_targets);random_edges=sorted(e for e in edges if e[0] in targetset);lookup=set(random_edges)
    rng=np.random.default_rng(20261027);accepted=0
    for _ in range(len(random_edges)*5):
        a,b=rng.integers(0,len(random_edges),size=2);ga,pa=random_edges[a];gb,pb=random_edges[b]
        if ga==gb or pa==pb or (ga,pb) in lookup or (gb,pa) in lookup:continue
        lookup.remove((ga,pa));lookup.remove((gb,pb));lookup.add((ga,pb));lookup.add((gb,pa));random_edges[a]=(ga,pb);random_edges[b]=(gb,pa);accepted+=1
    null,null_degree=mechanism(all_targets,random_edges)
    assert np.array_equal(null_degree,degree)
    np.savez_compressed(PREP/'mechanism_randomized.npz',target_ids=np.array(all_targets),embedding=null,degree=null_degree)
    write_json(PREP/'mechanism_provenance.json',{'source_url':'https://reactome.org/download/current/Ensembl2Reactome.txt','download_date':'2026-10-26','sha256':sha(p),'human_edges':len(edges),'eligible_target_edges':len(random_edges),'accepted_degree_preserving_swaps':accepted,'unsigned_membership_does_not_define_expression_direction':True,'labels_or_external_perturbation_effects_used':False})
    donor_lock=json.loads((BASE/'prepared/donor_split_lock.json').read_text(encoding='utf-8'))['assignment']
    donors=sorted(donor_lock);contexts=['Rest','Stim48hr','Stim8hr'];targets=all_targets+['NTC']
    final=PREP/'donor_effects.npy';temp=PREP/'donor_counts_work.npy'
    if final.exists() and (PREP/'donor_rows.csv').exists():status('complete_cached');return
    with h5py.File(paths['pseudobulk'],'r') as f:
        obs=f['obs'];g=index(f['var']);target=column(obs['perturbed_gene_id']).astype(str);d=column(obs['donor_id']).astype(str);c=column(obs['culture_condition']).astype(str)
        ncells=np.asarray(column(obs['n_cells']),dtype=float);guide=column(obs['guide_type']).astype(str)
        # Do NOT use keep_effective_guides or keep_for_DE: those use outcomes.
        eligible=(ncells>=5)&np.isfinite(ncells)&np.isin(target,targets)&np.isin(d,donors)&np.isin(c,contexts)&np.isin(guide,['targeting','non-targeting'])
        tmap={t:i for i,t in enumerate(targets)};dmap={v:i for i,v in enumerate(donors)};cmap={v:i for i,v in enumerate(contexts)}
        group=np.array([tmap[t]*12+dmap[dd]*3+cmap[cc] if ok else -1 for t,dd,cc,ok in zip(target,d,c,eligible)],dtype=np.int32)
        n=len(targets)*12;counts=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(n,len(g)));counts[:]=0
        libraries=np.zeros(n,np.float64);samples=np.zeros(n,np.int32);cellcounts=np.zeros(n,np.float64)
        for start in range(0,len(target),128):
            end=min(start+128,len(target));wanted=np.flatnonzero(eligible[start:end])
            if len(wanted):
                block=rows(f['X'],start,end)
                if not np.isfinite(block).all() or (block<0).any():raise ValueError('Invalid raw counts')
                for j in wanted:
                    i=start+j;k=group[i];counts[k]+=block[j];libraries[k]+=block[j].sum(dtype=np.float64);samples[k]+=1;cellcounts[k]+=ncells[i]
            if start%4096==0:counts.flush();status('aggregating_donor_counts',source_rows_done=start,source_rows_total=len(target))
        counts.flush()
    ntc_base=tmap['NTC']*12
    if not (libraries[ntc_base:ntc_base+12]>0).all():raise ValueError('Missing raw-QC donor/context control')
    control=np.log2(1+np.asarray(counts[ntc_base:ntc_base+12],dtype=np.float64)/libraries[ntc_base:ntc_base+12,None]*1e6).astype(np.float32)
    records=[]
    for t in all_targets:
        for dd in donors:
            for cc in contexts:
                k=tmap[t]*12+dmap[dd]*3+cmap[cc];z=dmap[dd]*3+cmap[cc]
                if libraries[k]>0:
                    counts[k]=np.log2(1+counts[k]/libraries[k]*1e6)-control[z]
                    records.append({'index':f'{t}|{dd}|{cc}','target_contrast':t,'culture_condition':cc,'donor_id':dd,'partition':donor_lock[dd],'storage_row':k,'n_pseudobulk_rows':int(samples[k]),'n_cells':float(cellcounts[k]),'effect_unit':'delta_log2_CPM_plus_1'})
                else:counts[k]=np.nan
        if tmap[t]%500==0:counts.flush();status('transforming_donor_effects',targets_done=tmap[t],targets_total=len(all_targets))
    counts[ntc_base:ntc_base+12]=np.nan;counts.flush();del counts;temp.replace(final)
    pd.DataFrame(records).to_csv(PREP/'donor_rows.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'gene_id':g}).to_csv(PREP/'donor_genes.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(PREP/'controls.npz',donors=np.array(donors),contexts=np.array(contexts),gene_ids=g,control_log2cpm=control,library_sizes=libraries[ntc_base:ntc_base+12])
    write_json(PREP/'donor_label_provenance.json',{'source_receipt':json.loads(paths['pseudobulk'].with_name(paths['pseudobulk'].name+'.verified.json').read_text()),'unit':'log2(1+CPM_target)-log2(1+CPM_matched_NTC)','normalization_library':'sum over all 18129 published genes before transformation','gene_universe':'all published pseudobulk reference genes; no pooled DE HVG subset','sample_filter':'n_cells >= 5; nonnegative finite counts; positive aggregated library; no effect/significance filtering','not_used':['keep_effective_guides','keep_for_DE','keep_test_genes','pooled_DE_effects'],'precision':'float32 accumulation and transformed cache; library totals float64','donor_assignment':donor_lock,'rows':len(records),'targets':len(all_targets),'genes':len(g),'limitation':'published curated target annotations and primary target coverage universe retained; not raw re-alignment or re-curation'})
    status('complete',donor_rows=len(records),genes=len(g))
if __name__=='__main__':main()
