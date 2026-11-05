"""First measured P2 experiment: simple baselines and shared/context residual.

No old NPZ, S9 significance-selected labels, or global-unseen-target claim.
The three context programs form a residual of rank at most three. This is a
minimal additive model, not the full proposed FVcell method or a SOTA claim.
"""
from pathlib import Path
import sys,os,json,hashlib,datetime
from atomic_json import write_json
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vendor'))
import numpy as np
import pandas as pd
import h5py
OUT=BASE/'training_first';OUT.mkdir(exist_ok=True)

def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(name,obj):
    write_json(OUT/name,obj)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def decode(values):
    return np.array([x.decode() if isinstance(x,bytes) else x for x in values])
def column(node):
    if isinstance(node,h5py.Dataset):return decode(node[:])
    if 'categories' in node and 'codes' in node:
        cats=column(node['categories']);codes=node['codes'][:]
        return np.array([cats[c] if c>=0 else None for c in codes],dtype=object)
    raise ValueError('Unsupported dataframe column encoding: '+node.name)
def index(group):
    key=group.attrs.get('_index','_index')
    if isinstance(key,bytes):key=key.decode()
    return column(group[key]).astype(str)
def matrix_shape(node):
    return tuple(int(v) for v in node.shape) if isinstance(node,h5py.Dataset) else tuple(int(v) for v in node.attrs['shape'])
def rows(node,start,end):
    if isinstance(node,h5py.Dataset):return np.asarray(node[start:end],dtype=np.float32)
    enc=node.attrs.get('encoding-type','')
    if isinstance(enc,bytes):enc=enc.decode()
    if enc!='csr_matrix':raise ValueError('Sparse matrix must be CSR; refuse an implicit conversion of a huge file')
    ptr=node['indptr'][start:end+1];lo,hi=int(ptr[0]),int(ptr[-1])
    data=node['data'][lo:hi];ids=node['indices'][lo:hi]
    result=np.zeros((end-start,matrix_shape(node)[1]),dtype=np.float32)
    for i in range(end-start):
        a,b=int(ptr[i]-lo),int(ptr[i+1]-lo)
        np.add.at(result[i],ids[a:b],data[a:b])
    return result
def mean_finite(y,axis=0):
    mask=np.isfinite(y);return np.divide(np.where(mask,y,0).sum(axis=axis,dtype=np.float64),mask.sum(axis=axis),out=np.full(np.asarray(mask.sum(axis=axis)).shape,np.nan),where=mask.sum(axis=axis)>0).astype(np.float32)

def control_metadata(obs):
    # Explicit mapping verified against the completed public pseudobulk file.
    target_source='perturbed_gene_id' if 'perturbed_gene_id' in obs else 'target'
    mapping={'target':target_source,'culture_condition':'culture_condition','donor_id':'donor_id'}
    if not all(v in obs for v in mapping.values()):raise ValueError('Unrecognised pseudobulk annotation schema')
    meta=pd.DataFrame({k:column(obs[v]) for k,v in mapping.items()})
    controls=meta.target.astype(str).isin(['NO-TARGET','NO_TARGET','NTC'])
    if target_source=='perturbed_gene_id':
        if 'guide_type' not in obs or 'keep_for_DE' not in obs:raise ValueError('Actual pseudobulk control/QC fields missing')
        non_targeting=column(obs['guide_type']).astype(str)=='non-targeting'
        if not np.array_equal(non_targeting,controls.to_numpy()):raise ValueError('NTC target annotation disagrees with non-targeting guide type')
        keep=column(obs['keep_for_DE'])
        if keep.dtype.kind!='b':raise ValueError('keep_for_DE must be boolean; refuse string coercion')
        controls=controls & keep
    if not controls.any():raise ValueError('Author non-targeting control not resolved after author QC')
    return meta,controls,mapping

def fit_shared(y,target_codes,train_mask):
    n=int(target_codes.max())+1;g=y.shape[1]
    sums=np.zeros((n,g),np.float32);counts=np.zeros((n,g),np.uint8)
    for i in np.flatnonzero(train_mask):
        mask=np.isfinite(y[i]);sums[target_codes[i],mask]+=y[i,mask];counts[target_codes[i],mask]+=1
    return sums,counts
def shared_row(y,i,target_codes,train_mask,sums,counts,global_mean,leave_self=False):
    t=target_codes[i];s=sums[t].copy();n=counts[t].astype(np.int16)
    if leave_self and train_mask[i]:
        finite=np.isfinite(y[i]);s[finite]-=y[i,finite];n[finite]-=1
    result=np.divide(s,n,out=global_mean.copy(),where=n>0)
    return result
def metrics(y,indices,predict,contexts,partition):
    records=[]
    for i in indices:
        pred=predict(i);true=y[i];mask=np.isfinite(true)
        if not np.isfinite(pred[mask]).all():raise ValueError('Nonfinite prediction for an observed response')
        if not mask.any():raise ValueError('No finite evaluation labels for row '+str(i))
        a=true[mask].astype(np.float64);b=pred[mask].astype(np.float64)
        corr=float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None
        direction=np.abs(a)>0.1
        records.append({'row':int(i),'partition':partition,'context':str(contexts[i]),'observed_genes':int(mask.sum()),'mae':float(np.abs(a-b).mean()),'pearson':corr,'direction_accuracy_abs_truth_gt_0_1':float((np.sign(a[direction])==np.sign(b[direction])).mean()) if direction.any() else None})
    frame=pd.DataFrame(records)
    return {'macro_context_target_mae':float(frame.groupby('context').mae.mean().mean()),'by_context_mae':{str(k):float(v) for k,v in frame.groupby('context').mae.mean().items()},'rows':len(frame)},frame

def main():
    lock=OUT/'training.lock'
    if (OUT/'results.json').exists():print('First experiment already complete; no duplicate run');return
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        save('status.json',{'state':'validating_inputs','started':stamp(),'pid':os.getpid()})
        frozen=json.loads((BASE/'prepared/FREEZE_LOCK.json').read_text(encoding='utf-8'))
        for name,value in frozen['sha256'].items():
            if sha(BASE/'prepared'/name)!=value:raise ValueError('Frozen protocol changed: '+name)
        manifest=json.loads((BASE/'large_asset_manifest.json').read_text(encoding='utf-8'))
        paths={};receipts={}
        for key in ('de_stats','pseudobulk'):
            p=BASE/'large_data'/Path(manifest[key]['key']).name
            receipt=json.loads(p.with_name(p.name+'.verified.json').read_text(encoding='utf-8'))
            if p.stat().st_size!=manifest[key]['bytes'] or receipt['etag']!=manifest[key]['etag'] or sha(p)!=receipt['sha256']:raise ValueError('Invalid source receipt: '+key)
            paths[key]=p;receipts[key]=receipt
        splits=pd.read_csv(BASE/'prepared/condition_and_global_target_splits.csv').set_index('index')
        if not splits.index.is_unique:raise ValueError('Duplicate split ID')
        with h5py.File(paths['pseudobulk'],'r') as p:
            obs=p['obs'];meta,controls,control_mapping=control_metadata(obs)
            if not set(splits.culture_condition.unique())<=set(meta.loc[controls,'culture_condition']):raise ValueError('Missing control context')
            control_audit=meta.loc[controls].groupby(['culture_condition','donor_id','target']).size().reset_index(name='pseudobulk_rows')
            control_audit.to_csv(OUT/'control_coverage.csv',index=False,encoding='utf-8-sig')
            if 'X' not in p:raise ValueError('Missing pseudobulk count matrix')
            # Check actual control count rows without loading the 44GB matrix.
            for i in np.flatnonzero(controls)[:32]:
                v=rows(p['X'],int(i),int(i)+1)
                if not np.isfinite(v).all() or (v<0).any():raise ValueError('Invalid control counts')
        with h5py.File(paths['de_stats'],'r') as f:
            ids=index(f['obs']);genes=index(f['var'])
            if len(set(ids))!=len(ids) or len(set(genes))!=len(genes):raise ValueError('Nonunique observation/gene IDs')
            if 'log_fc' not in f['layers']:raise ValueError('Published log2FC layer log_fc not found; do not guess another layer')
            node=f['layers/log_fc']
            if matrix_shape(node)!=(len(ids),len(genes)):raise ValueError('Response matrix dimensions disagree')
            if not set(splits.index)<=set(ids):raise ValueError('Split observations missing from data')
            selected=np.array([i for i,x in enumerate(ids) if x in splits.index],dtype=int)
            selected_ids=ids[selected];s=splits.loc[selected_ids].reset_index()
            for key in ('culture_condition','target_contrast'):
                if key not in f['obs'] or not np.array_equal(column(f['obs'][key])[selected].astype(str),s[key].astype(str).to_numpy()):raise ValueError('Observation annotation mismatch: '+key)
            gene_path=OUT/'genes.csv';pd.DataFrame({'gene_id':genes}).to_csv(gene_path,index=False,encoding='utf-8-sig')
            y=np.lib.format.open_memmap(OUT/'effects.npy',mode='w+',dtype=np.float32,shape=(len(selected),len(genes)))
            pos={int(x):i for i,x in enumerate(selected)};finite_total=0
            for start in range(0,len(ids),128):
                end=min(start+128,len(ids));wanted=[i for i in range(start,end) if i in pos]
                if wanted:
                    block=rows(node,start,end)
                    for i in wanted:
                        y[pos[i]]=block[i-start];finite_total+=int(np.isfinite(block[i-start]).sum())
            y.flush()
        if not finite_total:raise ValueError('No finite measured responses')
        validation={'status':'validated','time':stamp(),'source_receipts':receipts,'freeze_hash':sha(BASE/'prepared/FREEZE_LOCK.json'),'gene_id_sha256':sha(gene_path),'effect_layer':'log_fc','effect_unit':'published_DESeq2_log2FC','rows':len(y),'genes':len(genes),'finite_observations':finite_total,'control_values':sorted(meta.loc[controls,'target'].astype(str).unique().tolist()),'pseudobulk_column_mapping':control_mapping,'control_qc':'actual perturbed_gene_id=NTC matched to guide_type=non-targeting and keep_for_DE=true','counts_control_rows_checked':min(32,int(controls.sum())),'control_validation_scope':'author analysis and actual control context/donor coverage; full count matrix is not reanalysed','gene_selection_limitation':'author upstream feature selection retained; no new significance filtering'}
        save('input_validation.json',validation)
        save('status.json',{'state':'training_first_additive_model','time':stamp(),'pid':os.getpid()})
        train=(s.combo_partition=='train').to_numpy();val=np.flatnonzero(s.combo_partition=='validation');test=np.flatnonzero(s.combo_partition=='test')
        targets,target_codes=np.unique(s.target_contrast.astype(str),return_inverse=True)
        contexts=s.culture_condition.astype(str).to_numpy();ctx_names,ctx_codes=np.unique(contexts,return_inverse=True)
        global_mean=mean_finite(y[train]);global_mean=np.nan_to_num(global_mean,nan=0.)
        sums,counts=fit_shared(y,target_codes,train)
        residual_sum=np.zeros((len(ctx_names),len(genes)),np.float64);residual_count=np.zeros_like(residual_sum,dtype=np.int32)
        for i in np.flatnonzero(train):
            shared=shared_row(y,i,target_codes,train,sums,counts,global_mean,True)
            residual=y[i]-shared;mask=np.isfinite(residual);c=ctx_codes[i]
            residual_sum[c,mask]+=residual[mask];residual_count[c,mask]+=1
        programs=np.divide(residual_sum,residual_count,out=np.zeros_like(residual_sum),where=residual_count>0).astype(np.float32)
        def shared(i):return shared_row(y,i,target_codes,train,sums,counts,global_mean)
        # Predeclared small grid; validation only. Test opened only after selection.
        choices=[]
        for shrink in (0.,0.25,0.5,1.):
            for strength in (0.,0.25,0.5,1.):
                pred=lambda i,a=shrink,b=strength:a*shared(i)+(1-a)*global_mean+b*programs[ctx_codes[i]]
                m,_=metrics(y,val,pred,contexts,'validation');choices.append({'shared_weight':shrink,'context_residual_weight':strength,**m})
        best=min(choices,key=lambda x:x['macro_context_target_mae'])
        save('validation_selection.json',{'selection_metric':'macro_context_target_mae','grid':choices,'selected':best,'test_used_for_selection':False})
        np.savez_compressed(OUT/'model.npz',context_programs=programs,global_training_mean=global_mean,context_names=ctx_names,target_ids=targets,training_target_sums=sums,training_target_counts=counts,shared_weight=best['shared_weight'],context_residual_weight=best['context_residual_weight'])
        predictors={'zero_response':lambda i:np.zeros(len(genes),np.float32),'training_global_mean':lambda i:global_mean,'same_target_training_context_mean_P2':shared,'shared_plus_context_residual_P2':lambda i:best['shared_weight']*shared(i)+(1-best['shared_weight'])*global_mean+best['context_residual_weight']*programs[ctx_codes[i]],'context_residual_ablation':lambda i:best['shared_weight']*shared(i)+(1-best['shared_weight'])*global_mean}
        results={}
        for name,predict in predictors.items():
            results[name],per_row=metrics(y,test,predict,contexts,'test')
            per_row['observation_id']=s.loc[per_row.row,'index'].to_numpy()
            per_row['target_id']=s.loc[per_row.row,'target_contrast'].to_numpy()
            per_row.to_csv(OUT/(name+'_test_rows.csv'),index=False,encoding='utf-8-sig')
        save('results.json',{'time':stamp(),'task':'P2_heldout_target_context_combination','model':'minimal_additive_shared_response_plus_context_programs_rank_at_most_3','main_metric':'unweighted_signed_log2FC_MAE_macro_context_then_target','selected_hyperparameters':best,'test_results':results,'limitations':['first experiment only; not full FVcell method or SOTA comparison','no unseen-donor, unseen-target, organ, functional-validation or dynamics claim','author upstream gene selection remains a limitation','all finite published effects including zeros and nonsignificant values retained'],'training_only_fit':True})
        save('status.json',{'state':'complete','time':stamp(),'results_file':str(OUT/'results.json')})
    except BaseException as error:
        save('status.json',{'state':'blocked_or_failed','time':stamp(),'error':str(error)});raise
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
