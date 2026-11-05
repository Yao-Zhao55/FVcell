"""Shared, auditable training code for the outline's feasible experiments."""
from pathlib import Path
import sys,hashlib,json,time,copy,os
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'full_experiments';PREP=OUT/'prepared'
sys.path.insert(0,str(BASE/'vendor_state'));sys.path.insert(0,str(BASE/'vendor_ml'))
from run_first_training import sha
from atomic_json import write_json
import numpy as np,pandas as pd,torch
from torch import nn
torch.set_num_threads(6)

def hashnum(s):return int(hashlib.sha256(str(s).encode()).hexdigest()[:8],16)
def measured(y,meta,indices):
    storage=meta.iloc[indices].storage_row.to_numpy() if 'storage_row' in meta else np.asarray(indices)
    return np.asarray(y[storage],dtype=np.float32)

def features(y,meta,genes,permission='P2',mechanism_kind='real',train_mask=None):
    train=(meta.partition=='train').to_numpy() if train_mask is None else train_mask.copy()
    if not train.any():raise ValueError('No training labels')
    targets,codes=np.unique(meta.target_contrast.astype(str),return_inverse=True)
    mpath=PREP/('mechanism_randomized.npz' if mechanism_kind=='randomized' else 'mechanism.npz')
    m=np.load(mpath,allow_pickle=False);lookup={v:i for i,v in enumerate(m['target_ids'])}
    mech=np.stack([m['embedding'][lookup[t]] if t in lookup else np.random.default_rng(hashnum(t)).normal(size=32)/np.sqrt(32) for t in targets]).astype(np.float32)
    if mechanism_kind=='id_random':mech=np.stack([np.random.default_rng(hashnum('null|'+t)).normal(size=32)/np.sqrt(32) for t in targets]).astype(np.float32)
    ctrl=np.load(PREP/'controls.npz',allow_pickle=False);gmap={g:i for i,g in enumerate(ctrl['gene_ids'])}
    if not set(genes)<=set(gmap):raise ValueError('Control gene universe mismatch')
    positions=np.array([gmap[g] for g in genes]);control=ctrl['control_log2cpm'][:,positions]
    contexts=sorted(meta.culture_condition.unique());cmap={c:i for i,c in enumerate(contexts)}
    ccode=meta.culture_condition.map(cmap).to_numpy();control_index={(d,c):i*3+j for i,d in enumerate(ctrl['donors']) for j,c in enumerate(ctrl['contexts'])}
    rng=np.random.default_rng(20261026);projection=rng.normal(size=(len(genes),32)).astype(np.float32)/np.sqrt(len(genes))
    z=np.zeros((len(meta),35),np.float32)
    overrides=meta.attrs.get('control_override')
    if permission!='P0':
        ctrl_latent=control@projection
        extra={k:np.asarray(v,dtype=np.float32)@projection for k,v in overrides.items()} if overrides is not None else {}
        for i,r in enumerate(meta.itertuples()):
            key=(r.donor_id,r.culture_condition) if hasattr(r,'donor_id') else None
            if key in extra:z[i,:32]=extra[key]
            elif hasattr(r,'donor_id'):z[i,:32]=ctrl_latent[control_index[key]]
            else:z[i,:32]=np.mean([ctrl_latent[control_index[d,r.culture_condition]] for d in ctrl['donors']],axis=0)
    if meta.attrs.get('hashed_context_encoding'):
        context_vectors=np.stack([np.random.default_rng(hashnum('context|'+str(c))).normal(size=3) for c in contexts]).astype(np.float32)
        z[:,32:]=context_vectors[ccode]
    else:
        if len(contexts)>3:raise ValueError('More than three contexts require explicitly declared context encoding')
        z[np.arange(len(meta)),32+ccode]=1
    sums=np.zeros((len(targets),len(genes)),np.float32);counts=np.zeros(len(targets),np.int32)
    global_sum=np.zeros(len(genes),np.float64);global_sq=np.zeros(len(genes),np.float64);ntrain=0
    train_indices=np.flatnonzero(train)
    for start in range(0,len(train_indices),256):
        ii=train_indices[start:start+256];block=measured(y,meta,ii)
        if not np.isfinite(block).all():raise ValueError('Full-suite labels contain missing/nonfinite entries; explicit mask adaptation required')
        for j,i in enumerate(ii):sums[codes[i]]+=block[j];counts[codes[i]]+=1
        global_sum+=block.sum(axis=0,dtype=np.float64);global_sq+=np.square(block.astype(np.float64)).sum(axis=0);ntrain+=len(ii)
    mean=(global_sum/ntrain).astype(np.float32);std=np.sqrt(np.maximum(global_sq/ntrain-mean.astype(float)**2,0)).astype(np.float32)
    evidence_mask=np.asarray(meta.attrs.get('evidence_mask',train),dtype=bool)
    if permission=='P2' and not np.array_equal(evidence_mask,train):
        if evidence_mask.shape!=train.shape or (train&~evidence_mask).any():raise ValueError('Invalid strict evidence mask')
        if (evidence_mask & meta.partition.isin(['validation','test']).to_numpy()).any():raise ValueError('Held response included in numeric evidence')
        sums[:]=0;counts[:]=0
        evidence_indices=np.flatnonzero(evidence_mask)
        for start in range(0,len(evidence_indices),256):
            ii=evidence_indices[start:start+256];block=measured(y,meta,ii)
            if not np.isfinite(block).all():raise ValueError('Nonfinite explicit response evidence')
            for j,i in enumerate(ii):sums[codes[i]]+=block[j];counts[codes[i]]+=1
    u=np.zeros((len(meta),65),np.float32);u[:,:32]=mech[codes]
    if permission=='P2':
        sumproj=sums@projection;meanproj=mean@projection
        for i in range(len(meta)):
            t=codes[i];u[i,32:64]=sumproj[t]/counts[t] if counts[t] else meanproj;u[i,64]=float(counts[t]>0)
        for start in range(0,len(train_indices),256):
            ii=train_indices[start:start+256];p=measured(y,meta,ii)@projection
            for j,i in enumerate(ii):
                t=codes[i];u[i,32:64]=(sumproj[t]-p[j])/(counts[t]-1) if counts[t]>1 else meanproj
                u[i,64]=float(counts[t]>1)
    # Fit transforms on training rows only. P0/P1 never project held response labels.
    transform={}
    for name,a in [('u',u),('z',z)]:
        mu=a[train].mean(axis=0);scale=a[train].std(axis=0);scale=np.where(scale>.01,scale,1)
        transform[name+'_mean']=mu;transform[name+'_scale']=scale;a-=mu;a/=scale
    return {'u':u,'z':z,'context_codes':ccode,'mean':mean,'std':std,'target_sums':sums,'target_counts':counts,'target_codes':codes,'target_ids':targets,'mechanism':mech[codes],'train_mask':train,'transform':transform,'projection':projection}

class Factorized(nn.Module):
    def __init__(self,g,kind='fv',freeze_state=False):
        super().__init__();self.kind=kind;self.frozen=freeze_state
        self.shared=nn.Sequential(nn.Linear(65,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,32))
        self.action=nn.Linear(65,32)
        self.state=nn.Linear(35,32)
        self.shared_out=nn.Linear(32,g);self.residual_out=nn.Linear(32,g,bias=False)
        # Start close to training mean; learn a signed response, no output ReLU.
        nn.init.normal_(self.shared_out.weight,std=.001);nn.init.zeros_(self.shared_out.bias);nn.init.normal_(self.residual_out.weight,std=.001)
        if freeze_state:
            for p in self.state.parameters():p.requires_grad=False
    def forward(self,u,z,c=None):
        shared=self.shared(u);r=self.action(u)*torch.tanh(self.state(z))
        pred=self.shared_out(shared)+self.residual_out(r)
        return pred,r,shared

class Concatenation(nn.Module):
    def __init__(self,g,kind='concat'):
        super().__init__();self.kind=kind
        self.enc=nn.Sequential(nn.Linear(100,64),nn.GELU(),nn.Dropout(.1))
        self.out=nn.Linear(64,g);nn.init.normal_(self.out.weight,std=.001);nn.init.zeros_(self.out.bias)
    def forward(self,u,z,c=None):
        if self.kind=='shared':z=torch.zeros_like(z)
        h=self.enc(torch.cat([u,z],dim=-1));return self.out(h),h,h

class Independent(nn.Module):
    def __init__(self,g,ncontexts=3):
        super().__init__();self.models=nn.ModuleList([nn.Sequential(nn.Linear(100,21),nn.GELU(),nn.Dropout(.1),nn.Linear(21,g)) for _ in range(ncontexts)])
        for m in self.models:nn.init.normal_(m[-1].weight,std=.001);nn.init.zeros_(m[-1].bias)
    def forward(self,u,z,c):
        x=torch.cat([u,z],dim=-1);out=torch.zeros((len(u),self.models[0][-1].out_features),device=u.device)
        for k,m in enumerate(self.models):
            mask=c==k
            if mask.any():out[mask]=m(x[mask])
        return out,x,x

def build_model(family,g):
    if family in ('fv','no_mechanism','randomized_mechanism','no_shrink','fixed_state','no_mechanism_regularizer','onehot_state'):return Factorized(g,freeze_state=family=='fixed_state')
    if family in ('concat','shared'):return Concatenation(g,family)
    if family=='independent':return Independent(g)
    if family=='state_official':
        from full_suite_state_adapter import StateAdapter
        return StateAdapter(g)
    raise ValueError(family)

def scores(y,meta,indices,pred,save_to=None):
    if len(pred)!=len(indices):raise ValueError('Prediction row mismatch')
    records=[]
    for start in range(0,len(indices),128):
        ii=indices[start:start+128];true=measured(y,meta,ii).astype(np.float64);p=np.asarray(pred[start:start+len(ii)],dtype=np.float64)
        if not np.isfinite(true).all() or not np.isfinite(p).all():raise ValueError('Nonfinite evaluation')
        errors=np.abs(true-p).mean(axis=1);a=true-true.mean(axis=1,keepdims=True);b=p-p.mean(axis=1,keepdims=True);den=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1));corr=np.divide((a*b).sum(axis=1),den,out=np.full(len(ii),np.nan),where=den>0)
        dirs=np.abs(true)>.1;dcount=dirs.sum(axis=1);correct=((np.sign(true)==np.sign(p))&dirs).sum(axis=1);da=np.divide(correct,dcount,out=np.full(len(ii),np.nan),where=dcount>0)
        for j,i in enumerate(ii):
            r=meta.iloc[i];records.append({'index':r['index'],'target':r.target_contrast,'context':r.culture_condition,'donor':r.get('donor_id','pooled'),'mae':float(errors[j]),'pearson':float(corr[j]),'direction_accuracy':float(da[j]),'direction_genes':int(dcount[j])})
    f=pd.DataFrame(records)
    # Equal donor then context then target weight; primary has one pooled donor.
    agg=f.groupby(['donor','context']).mae.mean()
    summary={'mae':float(agg.groupby('donor').mean().mean()),'pearson':None if f.pearson.notna().sum()==0 else float(f.pearson.mean()),'direction_accuracy':None if f.direction_accuracy.notna().sum()==0 else float(f.direction_accuracy.mean()),'rows':len(f),'by_context_mae':{str(k):float(v) for k,v in f.groupby('context').mae.mean().items()},'by_donor_mae':{str(k):float(v) for k,v in f.groupby('donor').mae.mean().items()}}
    if save_to is not None:f.to_csv(save_to,index=False,encoding='utf-8-sig')
    return summary

def selection_mae(y,meta,indices,pred):
    """Same float64 MAE aggregation, without unused ranking/correlation work."""
    errors=np.empty(len(indices),np.float64)
    for start in range(0,len(indices),256):
        ii=indices[start:start+256];a=measured(y,meta,ii).astype(np.float64);b=np.asarray(pred[start:start+len(ii)],dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite selection labels/predictions')
        errors[start:start+len(ii)]=np.abs(a-b).mean(axis=1)
    frame=pd.DataFrame({'error':errors,'context':meta.iloc[indices].culture_condition.astype(str).to_numpy(),'donor':meta.iloc[indices].donor_id.astype(str).to_numpy() if 'donor_id' in meta else 'pooled'})
    return float(frame.groupby(['donor','context']).error.mean().groupby('donor').mean().mean())

@torch.no_grad()
def predict(model,f,indices,mean,device='cuda',batch=256):
    model.eval();out=np.empty((len(indices),len(mean)),np.float32)
    for start in range(0,len(indices),batch):
        ii=indices[start:start+batch];u=torch.tensor(f['u'][ii],device=device);z=torch.tensor(f['z'][ii],device=device);c=torch.tensor(f['context_codes'][ii],device=device)
        value=model(u,z,c)[0].float().cpu().numpy()+mean;out[start:start+len(ii)]=value
    return out

def fit(y,meta,f,family,seed,lr,run_dir,protocol):
    torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(family,y.shape[1]).to(device)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    train=np.flatnonzero(f['train_mask']);val=np.flatnonzero((meta.partition=='validation').to_numpy() & (meta.selection_or_calibration=='selection').to_numpy())
    if not len(val):raise ValueError('No selection validation rows')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    u=torch.tensor(f['u'],device=device);z=torch.tensor(f['z'],device=device);c=torch.tensor(f['context_codes'],device=device);anchor=torch.tensor(f['mechanism'],device=device)
    mean=torch.tensor(f['mean'],device=device)
    # Cache training rows only. Allocate before copying so competing workers
    # cannot both reserve a whole donor matrix; fall back safely to streaming.
    resident=None;resident_lookup=None;cache_start=time.time();cache_bytes=len(train)*y.shape[1]*4
    free_bytes=torch.cuda.mem_get_info()[0] if device=='cuda' else 0
    if device=='cuda' and cache_bytes<5_500_000_000 and free_bytes>cache_bytes+750_000_000:
        try:
            resident=torch.empty((len(train),y.shape[1]),device=device,dtype=torch.float32)
            for start in range(0,len(train),512):
                ii=train[start:start+512];resident[start:start+len(ii)].copy_(torch.from_numpy(measured(y,meta,ii)).to(device))
            lookup=np.full(len(meta),-1,np.int64);lookup[train]=np.arange(len(train));resident_lookup=torch.tensor(lookup,device=device)
        except torch.cuda.OutOfMemoryError:
            resident=None;resident_lookup=None;torch.cuda.empty_cache()
    cache_seconds=time.time()-cache_start
    best=float('inf');best_state=None;stale=0;step=0;history=[];start_time=time.time();rng=np.random.default_rng(seed)
    while step<protocol['max_steps']:
        order=rng.permutation(train)
        for start in range(0,len(order),protocol['batch_size']):
            ii=order[start:start+protocol['batch_size']];ti=torch.tensor(ii,device=device)
            true=resident[resident_lookup[ti]] if resident is not None else torch.tensor(measured(y,meta,ii),device=device)
            model.train();optimizer.zero_grad(set_to_none=True)
            pred,residual,latent=model(u[ti],z[ti],c[ti]);pred=pred+mean
            loss=torch.nn.functional.smooth_l1_loss(pred,true,beta=.1)
            if family not in ('concat','shared','independent','state_official'):
                if family!='no_shrink':loss=loss+protocol['residual_penalty']*residual.square().mean()
                if family not in ('no_mechanism','no_mechanism_regularizer'):
                    # Annotation geometry constrains latent programs, never expression signs.
                    a=torch.nn.functional.normalize(latent,dim=-1);b=torch.nn.functional.normalize(anchor[ti],dim=-1)
                    loss=loss+protocol['mechanism_penalty']*((a@a.T)-(b@b.T)).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
            if step%protocol['validation_interval']==0 or step==protocol['max_steps']:
                vp=predict(model,f,val,f['mean'],device);metric=selection_mae(y,meta,val,vp);history.append({'step':step,'validation_mae':metric,'training_loss':float(loss.detach())})
                write_json(run_dir/'progress.json',{'state':'training','family':family,'seed':seed,'lr':lr,'step':step,'validation_mae':metric,'best_validation_mae':min(metric,best),'elapsed_seconds':time.time()-start_time})
                if metric<best-1e-6:best=metric;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                if stale>=protocol['patience']:break
            if step>=protocol['max_steps']:break
        if stale>=protocol['patience']:break
    if best_state is None:raise ValueError('No validated checkpoint')
    model.load_state_dict(best_state)
    return model,{'seed':seed,'lr':lr,'best_selection_mae':best,'steps':step,'elapsed_seconds':time.time()-start_time,'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'history':history,'device':device,'training_label_cache':'training_rows_only_on_GPU' if resident is not None else 'disk_backed_batch_streaming','cache_setup_seconds':cache_seconds,'peak_memory_measurement':'per-fit peak reset','peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device=='cuda' else None}
