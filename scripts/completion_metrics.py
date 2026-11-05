"""Vectorized metadata handling; the numerical metric definitions stay unchanged."""
from full_suite_core import np,pd,measured

def fast_scores(y,meta,indices,pred,save_to=None):
    pieces=[]
    for start in range(0,len(indices),128):
        ii=indices[start:start+128];a=measured(y,meta,ii).astype(np.float64);p=np.asarray(pred[start:start+len(ii)],dtype=np.float64)
        assert np.isfinite(a).all() and np.isfinite(p).all();error=np.abs(a-p).mean(1);aa=a-a.mean(1,keepdims=True);bb=p-p.mean(1,keepdims=True);den=np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1));corr=np.divide((aa*bb).sum(1),den,out=np.full(len(ii),np.nan),where=den>0)
        direction=np.abs(a)>.1;count=direction.sum(1);correct=((np.sign(a)==np.sign(p))&direction).sum(1);da=np.divide(correct,count,out=np.full(len(ii),np.nan),where=count>0);m=meta.iloc[ii]
        pieces.append(pd.DataFrame({'index':m['index'].astype(str).to_numpy(),'target':m.target_contrast.to_numpy(),'context':m.culture_condition.to_numpy(),'donor':m.donor_id.to_numpy() if 'donor_id' in m else np.full(len(ii),'pooled'),'mae':error,'pearson':corr,'direction_accuracy':da,'direction_genes':count}))
    f=pd.concat(pieces,ignore_index=True);agg=f.groupby(['donor','context']).mae.mean()
    summary={'mae':float(agg.groupby('donor').mean().mean()),'pearson':None if not f.pearson.notna().any() else float(f.pearson.mean()),'direction_accuracy':None if not f.direction_accuracy.notna().any() else float(f.direction_accuracy.mean()),'rows':len(f),'by_context_mae':{str(k):float(v) for k,v in f.groupby('context').mae.mean().items()},'by_donor_mae':{str(k):float(v) for k,v in f.groupby('donor').mae.mean().items()}}
    if save_to is not None:f.to_csv(save_to,index=False,encoding='utf-8-sig')
    return summary

def fast_calibration(y,meta,cal,cp,test,tp,scale):
    scale=np.maximum(scale,.05);summary={}
    for ctx in sorted(meta.culture_condition.unique()):
        cj=np.flatnonzero(meta.iloc[cal].culture_condition.to_numpy()==ctx);tj=np.flatnonzero(meta.iloc[test].culture_condition.to_numpy()==ctx)
        if not len(cj) or not len(tj):continue
        s=[]
        for start in range(0,len(cj),128):
            jj=cj[start:start+128];err=np.abs(measured(y,meta,cal[jj])-cp[jj]);s.extend((err/scale).max(1).astype(float).tolist())
        n=len(s);rank=int(np.ceil((n+1)*.9));q=float(np.sort(s)[min(rank,n)-1]);covered=0;point=0;total=0
        for start in range(0,len(tj),128):
            jj=tj[start:start+128];ok=np.abs(measured(y,meta,test[jj])-tp[jj])<=scale*q;covered+=int(ok.all(1).sum());point+=int(ok.sum());total+=ok.size
        summary[str(ctx)]={'calibration_rows':n,'calibration_order_stat_rank':rank,'simultaneous_response_vector_coverage':covered/len(tj),'coordinate_coverage':point/total,'mean_interval_width':float(np.mean(2*scale*q)),'nominal':.9}
    return {'by_context':summary,'guarantee':'empirical coverage only; cross-condition/target dependence and donor shift do not establish exchangeability','calibration_targets_not_used_for_early_stopping_or_model_selection':True}
