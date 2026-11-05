"""Inspect and aggregate published external raw counts; refuse normalized inputs."""
from pathlib import Path
import json,re,time
from full_suite_core import BASE,OUT,PREP,np,pd,write_json
from run_first_training import h5py,column,index,rows,matrix_shape
DEST=OUT/'external_data'

def inspect():
    result={}
    for path in DEST.glob('*.h5ad'):
        if not path.with_suffix('.receipt.json').exists():continue
        with h5py.File(path,'r') as f:
            info={'obs':list(f['obs'].keys()),'var':list(f['var'].keys()),'var_index_examples':index(f['var'])[:5].tolist(),'layers':list(f['layers'].keys()),'X_shape':matrix_shape(f['X']),'examples':{}}
            for key in ('perturbation','condition','gene','guide_id','perturbation_type','celltype','batch','gemgroup'):
                if key in f['obs']:
                    v=column(f['obs'][key]).astype(str);u=list(dict.fromkeys(v));info['examples'][key]={'n_unique':len(u),'first':u[:8],'control_like_values':[x for x in u if any(k in x.lower() for k in ('control','non-target','ntc'))][:10]}
            result[path.name]=info
    write_json(DEST/'schema.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

def aggregate(path,name):
    targetfile=DEST/(name+'_pseudobulk.npz')
    if targetfile.exists():return
    with h5py.File(BASE/'large_data/GWCD4i.pseudobulk_merged.h5ad','r') as primary:
        pg=index(primary['var']);pn=column(primary['var']['gene_name']).astype(str)
    ref=pd.DataFrame({'id':pg,'name':pn});unique=ref.groupby('name').id.nunique();symbol2id={r['name']:r['id'] for r in ref[ref.name.isin(unique[unique==1].index)].to_dict('records')}
    with h5py.File(path,'r') as f:
        obs=f['obs'];var=f['var'];varidx=index(var)
        ids=None
        for key in ('ensembl_id','ensembl_gene_id','gene_ids','gene_id'):
            if key in var:
                v=column(var[key]).astype(str)
                if np.mean(np.char.startswith(v,'ENSG'))>.8:ids=np.array([x.split('.')[0] for x in v]);break
        if ids is None:
            if np.mean(np.char.startswith(varidx,'ENSG'))>.8:ids=np.array([x.split('.')[0] for x in varidx])
            else:ids=np.array([symbol2id.get(x,'UNMAPPED:'+x) for x in varidx])
        if len(set(ids))!=len(ids):raise ValueError('Duplicate external gene IDs require explicit aggregation')
        labelkey='perturbation' if 'perturbation' in obs else 'condition' if 'condition' in obs else None
        if not labelkey:raise ValueError('External perturbation labels unresolved')
        labels=column(obs[labelkey]).astype(str);control_names={'control','ctrl','non-targeting','non_targeting','ntc','negative_control','non-targeting_control'}
        mapped=[]
        for value in labels:
            if value.lower() in control_names:mapped.append('NTC')
            elif value.startswith('ENSG'):mapped.append(value.split('.')[0])
            elif value in symbol2id:mapped.append(symbol2id[value])
            else:
                clean=re.sub(r'[_-]\d+$','',value);mapped.append(symbol2id.get(clean,'UNMAPPED:'+value))
        mapped=np.array(mapped);groups=sorted(set(mapped)-{x for x in mapped if x.startswith('UNMAPPED:')});lookup={x:i for i,x in enumerate(groups)}
        if 'NTC' not in lookup:raise ValueError('No explicit external non-targeting control')
        node=f['layers/counts'] if 'counts' in f['layers'] else f['X']
        sums=np.zeros((len(groups),len(ids)),np.float32);n=np.zeros(len(groups),np.int32);libraries=np.zeros(len(groups),np.float64)
        for start in range(0,len(labels),256):
            end=min(start+256,len(labels));block=rows(node,start,end)
            if not np.isfinite(block).all() or (block<0).any() or not np.allclose(block,np.rint(block),atol=1e-4,rtol=0):raise ValueError('External input is not nonnegative integral raw counts')
            for j in range(end-start):
                t=mapped[start+j]
                if t in lookup:
                    k=lookup[t];sums[k]+=block[j];libraries[k]+=block[j].sum(dtype=np.float64);n[k]+=1
            if start%16384==0:write_json(DEST/'aggregation_status.json',{'source':name,'rows_done':start,'total_rows':len(labels),'updated_epoch':time.time()})
        np.savez_compressed(targetfile,target_ids=np.array(groups),gene_ids=ids,counts=sums,library_sizes=libraries,n_cells=n)
        write_json(DEST/(name+'_aggregation_provenance.json'),{'source_file':str(path),'source_receipt':json.loads(path.with_suffix('.receipt.json').read_text()),'label_column':labelkey,'counts_location':node.name,'controls_literal':[v for v in set(labels) if v.lower() in control_names],'raw_counts_integrality_checked_all_rows':True,'unmapped_observations':int(np.char.startswith(mapped,'UNMAPPED:').sum()),'single_cell_observations':len(labels),'aggregated_targets':len(groups)-1,'library_reference':'all retained original source genes, not intersection only','biological_unit':'cell line, not human donor'})

if __name__=='__main__':
    inspect()
    for filename,name in [('ReplogleWeissman2022_K562_essential.h5ad','K562'),('ReplogleWeissman2022_rpe1.h5ad','RPE1')]:
        path=DEST/filename
        if path.exists() and path.with_suffix('.receipt.json').exists():aggregate(path,name)
