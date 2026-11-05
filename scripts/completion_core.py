"""Prospectively specified additions; never modifies the earlier experiment suite."""
from full_suite_core import *
from full_suite_core import Factorized,Concatenation
import gc,itertools
ROOT=BASE/'training_completion';ROOT.mkdir(exist_ok=True)
PLAN={'version':'outline-completion-20261026-v1','seeds':[11,29,47],'lrs':[.001,.0003],'shrink_strengths':[.0003,.003],'steps':3000,'batch':128,'validate_every':250,'patience':4,'min_delta':1e-6,'selection':'validation selection half only; calibration half untouched','no_test_selection':True,'prior_exposure':'Earlier primary and donor test summaries have been seen. New additions are exploratory, not a pristine confirmation. No choices are made using new test results.','hierarchy':'Gaussian context and donor latent offsets with learned context scales partially pooled toward one global log scale; MAP, not full posterior inference','hierarchy_folds':'four fixed held-donor folds; offsets for unobserved donors are exactly zero; prior weighting never uses cell count as independent donors','missingness':'fixed artificial training-label deletion, 70% or 30% observed; all test responses remain real and complete; no biological synthetic responses','context':'Entire held culture removed from supervision and selection, global test targets absent from ALL supervision. P0 has no control expression; P1 permits NTC. Fixed stimulus metadata, no unseen ID embedding.','unchanged_old_sources':'all source labels, prior files, old runs and frozen splits read-only'}
def freeze():
    p=ROOT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==PLAN
    else:write_json(p,PLAN)
    return PLAN

class ObservedArray:
    def __init__(self,y,mask):self.y=y;self.mask=mask;self.shape=y.shape
    def __getitem__(self,idx):return np.where(self.mask[idx],self.y[idx],0.).astype(np.float32)

def mask_features(y,m,g,mask):
    proxy=ObservedArray(y,mask);f=features(proxy,m,g,'P1');ii=np.flatnonzero(m.partition.eq('train'));s=np.zeros(y.shape[1],np.float64);ss=s.copy();n=s.copy()
    for start in range(0,len(ii),256):
        ids=ii[start:start+256];a=proxy[ids].astype(np.float64);q=mask[ids];s+=a.sum(0);ss+=(a*a).sum(0);n+=q.sum(0)
    assert (n>0).all(),'An output gene has no measured training labels'
    mean=s/n;f['mean']=mean.astype(np.float32);f['std']=np.sqrt(np.maximum(ss/n-mean**2,0)).astype(np.float32)
    return proxy,f

def observed_mask_row(row_id,genes,rate):
    # Nested masks: every 30%-observed entry also appears in the 70% mask.
    return np.random.default_rng(hashnum('observed|'+str(row_id))).random(genes)<rate

def make_mask(y,m,rate):
    dest=ROOT/'masks';dest.mkdir(exist_ok=True);name=str(int(rate*100))
    p=dest/('observed_'+name+'.npy')
    if p.exists():return np.unpackbits(np.load(p),axis=1,count=y.shape[1]).astype(bool)
    q=np.ones(y.shape,bool)
    for i in np.flatnonzero(m.partition.eq('train')):q[i]=observed_mask_row(m.iloc[i]['index'],y.shape[1],rate)
    np.save(p,np.packbits(q,axis=1));write_json(dest/('observed_'+name+'.json'),{'rate':rate,'actual_training_observed':float(q[m.partition.eq('train')].mean()),'packed_sha256':sha(p),'mask_role':'artificial training missingness only, not a biological response dataset'})
    return q

def context_features(y,m,g,permission):
    f=features(y,m,g,permission);raw=f['z']*f['transform']['z_scale']+f['transform']['z_mean']
    hours=m.culture_condition.map({'Rest':0.,'Stim8hr':8.,'Stim48hr':48.}).to_numpy();assert np.isfinite(hours).all()
    raw[:,32]=hours>0;raw[:,33]=np.log1p(hours)/np.log(49.);raw[:,34]=1.
    tr=m.partition.eq('train').to_numpy();mu=raw[tr].mean(0);sd=raw[tr].std(0);sd=np.where(sd>.01,sd,1)
    f['z']=(raw-mu)/sd;f['transform'].update({'z_mean':mu,'z_scale':sd})
    return f

class Hierarchical(Factorized):
    def __init__(self,g,donors,kind):
        super().__init__(g);self.mode=kind;self.context_offset=nn.Parameter(torch.zeros(3,32));self.donor_offset=nn.Parameter(torch.zeros(donors,32))
        self.log_global=nn.Parameter(torch.tensor(-2.));self.log_context=nn.Parameter(torch.full((3,),-2.));self.log_donor=nn.Parameter(torch.tensor(-2.))
        if kind!='hierarchical':
            for p in (self.log_global,self.log_context,self.log_donor):p.requires_grad=False
    def forward(self,u,z,c,d=None):
        b=self.context_offset-self.context_offset.mean(0,keepdim=True);v=self.donor_offset-self.donor_offset.mean(0,keepdim=True)
        extra=torch.zeros((len(u),32),device=u.device);ok=d>=0
        if ok.any():extra[ok]=v[d[ok]]
        h=self.shared(u);r=self.action(u)*torch.tanh(self.state(z)+b[c]+extra)
        return self.shared_out(h)+self.residual_out(r),r,h
    def prior(self):
        if self.mode=='unshrunk_offsets':return self.context_offset.sum()*0
        b=self.context_offset-self.context_offset.mean(0,keepdim=True);v=self.donor_offset-self.donor_offset.mean(0,keepdim=True)
        lc=self.log_context.clamp(-6,1);ld=self.log_donor.clamp(-6,1);lg=self.log_global.clamp(-6,1)
        # Exact Gaussian normalization on the zero-sum subspace. For unequal
        # context scales, conditioning sum(b)=0 contributes -log(sum tau^2)/2;
        # D donor offsets have D-1, not D, independent degrees of freedom.
        p=(.5*(b.square()/torch.exp(2*lc[:,None])).sum()+b.shape[1]*(lc.sum()-.5*torch.logsumexp(2*lc,dim=0)))/b.numel()
        p=p+(.5*v.square().sum()/torch.exp(2*ld)+v.shape[1]*(v.shape[0]-1)*ld)/v.numel()
        if self.mode=='hierarchical':p=p+.5*(lc-lg).square().mean()+.5*(lg+2).square()+.5*(ld+2).square()
        return p

class IdentifiableHierarchy(Hierarchical):
    """Shared continuous state plus a uniquely regularized categorical path."""
    def forward(self,u,z,c,d=None):
        shared_z=torch.cat([z[:,:32],torch.zeros_like(z[:,32:])],dim=1)
        return super().forward(u,shared_z,c,d)

def convert_context_ID_path(model,f,meta):
    """Preserve the warm predictor, moving ALL one-hot effects into b_c."""
    tr=meta.partition.eq('train').to_numpy();effects=[]
    with torch.no_grad():
        for ctx in range(3):
            ii=np.flatnonzero(tr&(f['context_codes']==ctx));assert len(ii),'Conversion needs observed training contexts'
            a=f['z'][ii,32:];np.testing.assert_allclose(a,a[:1].repeat(len(a),axis=0),atol=1e-6)
            effects.append(torch.tensor(a[0],device=model.state.weight.device)@model.state.weight[:,32:].T)
        effect=torch.stack(effects);model.context_offset.add_(effect);model.state.bias.add_(effect.mean(0));model.state.weight[:,32:].zero_()

def model_for(family,g,donors):
    if family=='fv_continued':return Factorized(g)
    if family in ('hierarchical','fixed_gaussian','unshrunk_offsets'):return Hierarchical(g,donors,family)
    if family=='hierarchical_pooled':return Hierarchical(g,donors,'hierarchical')
    if family=='state_official':return build_model('state_official',g)
    return Factorized(g) if family=='fv' else Concatenation(g)

def forward(model,u,z,c,d):return model(u,z,c,d) if isinstance(model,Hierarchical) else model(u,z,c)

def observed_loss(pred,true,mask):
    if not bool(mask.any()):raise ValueError('Batch has no observed labels')
    if not bool(torch.isfinite(true[mask]).all()):raise ValueError('Observed label is nonfinite')
    a=torch.where(mask,pred,torch.zeros_like(pred));b=torch.where(mask,true,torch.zeros_like(true))
    return torch.nn.functional.smooth_l1_loss(a,b,beta=.1,reduction='sum')/mask.sum()

@torch.no_grad()
def prediction(model,f,idx,dc):
    model.eval();out=np.empty((len(idx),len(f['mean'])),np.float32)
    for start in range(0,len(idx),256):
        ii=idx[start:start+256];u=torch.tensor(f['u'][ii],device='cuda');z=torch.tensor(f['z'][ii],device='cuda');c=torch.tensor(f['context_codes'][ii],device='cuda');d=torch.tensor(dc[ii],device='cuda')
        out[start:start+len(ii)]=forward(model,u,z,c,d)[0].float().cpu().numpy()+f['mean']
    return out

def fit_new(y,m,f,family,seed,lr,strength,folder,dc,donors,mask=None,warm_checkpoint=None,max_steps=None):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    model=model_for(family,y.shape[1],donors).cuda();equivalence_error=None
    if warm_checkpoint is not None:
        model.load_state_dict(warm_checkpoint['state_dict'])
        if isinstance(model,Hierarchical):
            check=np.flatnonzero(m.partition.eq('train'))[:64];before=prediction(model,f,check,dc)
            convert_context_ID_path(model,f,m)
            corrected=IdentifiableHierarchy(y.shape[1],donors,model.mode).cuda();corrected.load_state_dict(model.state_dict());model=corrected
            after=prediction(model,f,check,dc);equivalence_error=float(np.max(np.abs(before-after)));assert equivalence_error<3e-5,equivalence_error
    budget=PLAN['steps'] if max_steps is None else max_steps
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    tr=np.flatnonzero(m.partition.eq('train'));vi=np.flatnonzero(m.partition.eq('validation')&m.selection_or_calibration.eq('selection'))
    u=torch.tensor(f['u'],device='cuda');z=torch.tensor(f['z'],device='cuda');c=torch.tensor(f['context_codes'],device='cuda');d=torch.tensor(dc,device='cuda');mean=torch.tensor(f['mean'],device='cuda');anchor=torch.tensor(f['mechanism'],device='cuda')
    resident=None;lookup=None;size=len(tr)*y.shape[1]*4;free=torch.cuda.mem_get_info()[0]
    if size<5_500_000_000 and free>size+800_000_000:
        try:
            resident=torch.empty((len(tr),y.shape[1]),device='cuda')
            for j in range(0,len(tr),512):resident[j:j+min(512,len(tr)-j)].copy_(torch.tensor(measured(y,m,tr[j:j+512]),device='cuda'))
            lookup=np.full(len(m),-1,np.int64);lookup[tr]=np.arange(len(tr));lookup=torch.tensor(lookup,device='cuda')
        except torch.cuda.OutOfMemoryError:resident=None;lookup=None;torch.cuda.empty_cache()
    start=time.time();best=float('inf');state=None;stale=0;step=0;history=[]
    while step<budget:
        for j in range(0,len(tr),PLAN['batch']):
            if j==0:order=rng.permutation(tr)
            ii=order[j:j+PLAN['batch']];ti=torch.tensor(ii,device='cuda');true=resident[lookup[ti]] if resident is not None else torch.tensor(measured(y,m,ii),device='cuda')
            if mask is not None:
                mi=m.iloc[ii].storage_row.to_numpy() if 'storage_row' in m else ii
                if not mask[mi].any():continue
            model.train();opt.zero_grad(set_to_none=True);p,r,h=forward(model,u[ti],z[ti],c[ti],d[ti]);p=p+mean
            if mask is not None:
                mi=m.iloc[ii].storage_row.to_numpy() if 'storage_row' in m else ii;q=torch.tensor(mask[mi],device='cuda');loss=observed_loss(p,true,q)
            else:loss=torch.nn.functional.smooth_l1_loss(p,true,beta=.1)
            if family not in ('concat','state_official'):
                loss=loss+.001*r.square().mean();a=torch.nn.functional.normalize(h,dim=-1);b=torch.nn.functional.normalize(anchor[ti],dim=-1);loss=loss+.001*((a@a.T)-(b@b.T)).square().mean()
            if isinstance(model,Hierarchical):loss=loss+strength*model.prior()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
            if step%PLAN['validate_every']==0 or step==budget:
                vp=prediction(model,f,vi,dc);v=selection_mae(y,m,vi,vp);history.append({'step':step,'validation_mae':v})
                write_json(folder/'progress.json',{'seed':seed,'lr':lr,'strength':strength,'step':step,'validation_mae':v,'elapsed_seconds':time.time()-start})
                if v<best-PLAN['min_delta']:best=v;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                if stale>=PLAN['patience']:break
            if step>=budget:break
        if stale>=PLAN['patience']:break
    assert state is not None;model.load_state_dict(state)
    return model,{'seed':seed,'lr':lr,'strength':strength,'best_selection_mae':best,'history':history,'steps':step,'seconds':time.time()-start,'parameters':sum(p.numel() for p in model.parameters()),'training_donors':donors,'training_labels_on_GPU':resident is not None,'warm_start':warm_checkpoint is not None,'training_only_reparameterization_max_error':equivalence_error,'categorical_path':'hierarchical_random_effect_only' if isinstance(model,IdentifiableHierarchy) else 'legacy_or_nonhierarchical'}

def run(name,y,m,g,families,permission='P1',mask=None,stimulus=False):
    freeze();task=ROOT/'runs'/name;task.mkdir(parents=True,exist_ok=True);m.to_csv(task/'rows.csv',index=False,encoding='utf8')
    if all((task/f/'summary.json').exists() for f in families):return
    tr=m.partition.eq('train');vi=m.partition.eq('validation');te=m.partition.eq('test');assert tr.any() and vi.any() and te.any()
    train_donors=sorted(m.loc[tr,'donor_id'].unique()) if 'donor_id' in m else ['pooled'];mapping={d:i for i,d in enumerate(train_donors)}
    dc=m.donor_id.map(mapping).fillna(-1).to_numpy(np.int64) if 'donor_id' in m else np.zeros(len(m),np.int64)
    source=y
    if mask is not None:y,f=mask_features(y,m,g,mask)
    else:f=context_features(y,m,g,permission) if stimulus else features(y,m,g,permission)
    ti=np.flatnonzero(te)
    write_json(task/'information_audit.json',{'permission':permission,'train_rows':int(tr.sum()),'test_rows':len(ti),'train_donors':train_donors,'held_donor_offsets':0,'training_only_scaling':True,'artificial_label_deletion':mask is not None,'stimulus_features':'fixed stimulated indicator, log1p(hours)/log49, constant; no learned unseen context ID' if stimulus else 'original NTC projection and context metadata','unit':'delta_log2_CPM_plus_1' if 'donor_id' in m else 'published_DESeq2_log2FC'})
    for family in families:
        dest=task/family;dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():continue
        write_json(ROOT/(os.environ.get('COMPLETION_BRANCH','work')+'_progress.json'),{'state':'running','task':name,'family':family,'updated_epoch':time.time()})
        if family in ('zero','training_mean'):
            p=np.zeros((len(ti),y.shape[1]),np.float32) if family=='zero' else np.tile(f['mean'],(len(ti),1));s=scores(source,m,ti,p,dest/'test_rows.csv');write_json(dest/'summary.json',{'deterministic':True,'test':s});continue
        if family=='ridge':
            from full_suite_run import ridge
            s=ridge(y,m,f,PLAN,dest);write_json(dest/'summary.json',{'deterministic':True,'test':s});continue
        results=[];selected=None
        for seed in PLAN['seeds']:
            folder=dest/str(seed);folder.mkdir(exist_ok=True);p=folder/'result.json'
            if p.exists():
                r=json.loads(p.read_text());results.append(r);selected=(r['fit']['lr'],r['fit']['strength']);continue
            strengths=[.001] if family=='hierarchical_pooled' else PLAN['shrink_strengths'] if family in ('hierarchical','fixed_gaussian') else [0.]
            trials=list(itertools.product(PLAN['lrs'],strengths)) if selected is None else [selected]
            best=None;trial_records=[]
            for lr,strength in trials:
                model,info=fit_new(y,m,f,family,seed,lr,strength,folder,dc,len(train_donors),mask)
                trial_records.append(info)
                if best is None or info['best_selection_mae']<best[1]['best_selection_mae']:best=({k:v.cpu().clone() for k,v in model.state_dict().items()},info)
                del model;gc.collect();torch.cuda.empty_cache()
            state,info=best;selected=(info['lr'],info['strength']);model=model_for(family,y.shape[1],len(train_donors)).cuda();model.load_state_dict(state);pval=prediction(model,f,ti,dc);s=scores(source,m,ti,pval,folder/'test_rows.csv')
            extra={}
            if isinstance(model,Hierarchical):extra={'log_context_scales':model.log_context.detach().cpu().tolist(),'log_global_scale':float(model.log_global.detach()),'log_donor_scale':float(model.log_donor.detach()),'inference':'MAP point estimate, not credible interval or full Bayesian posterior'}
            torch.save({'state_dict':state,'family':family,'genes':g.tolist(),'permission':permission,'mean':f['mean'],'feature_transform':f['transform'],'train_donor_mapping':mapping,'protocol_sha256':sha(ROOT/'protocol.json'),'fit':info},folder/'model.pt')
            result={'fit':info,'tuning_trials':trial_records,'test':s,'test_used_for_selection':False,'hierarchy':extra};write_json(folder/'result.json',result);results.append(result)
            if name=='hierarchical_functional_P1':np.save(folder/'test_predictions.npy',pval)
            del model,pval,state,best;gc.collect();torch.cuda.empty_cache()
        write_json(dest/'summary.json',{'seeds':PLAN['seeds'],'mean_test_MAE':float(np.mean([r['test']['mae'] for r in results])),'sd_test_MAE':float(np.std([r['test']['mae'] for r in results],ddof=1)),'runs':results})
    write_json(task/'complete.json',{'state':'complete_for_recorded_families','families':sorted(p.name for p in task.iterdir() if p.is_dir() and (p/'summary.json').exists()),'updated_epoch':time.time()})
    del f;gc.collect();torch.cuda.empty_cache()
