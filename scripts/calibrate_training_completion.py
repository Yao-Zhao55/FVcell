"""Reserved-validation calibration of new donor models; no parameter updates."""
from completion_core import *
from completion_metrics import fast_calibration as calibration

def main():
    y=np.load(PREP/'donor_effects.npy',mmap_mode='r');genes=pd.read_csv(PREP/'donor_genes.csv').gene_id.to_numpy();dest=ROOT/'calibration';dest.mkdir(exist_ok=True)
    for fold in range(4):
        task=ROOT/'runs'/('hierarchical_donor_fold'+str(fold));families=['fixed_gaussian','hierarchical','unshrunk_offsets','fv_continued']
        if all((dest/(str(fold)+'_'+family+'.json')).exists() for family in families):continue
        m=pd.read_csv(task/'rows.csv');assert all((task/f/'summary.json').exists() for f in families)
        f=features(y,m,genes,'P1');f.pop('target_sums',None);f.pop('target_counts',None)
        train=m.partition.eq('train');donors=sorted(m.loc[train,'donor_id'].unique());mapping={d:i for i,d in enumerate(donors)};dc=m.donor_id.map(mapping).fillna(-1).to_numpy(np.int64)
        ci=np.flatnonzero(m.partition.eq('validation')&m.selection_or_calibration.eq('calibration'));ti=np.flatnonzero(m.partition.eq('test'))
        for family in families:
            p=dest/(str(fold)+'_'+family+'.json')
            if p.exists():continue
            cpmean=None;tpmean=None
            for seed in PLAN['seeds']:
                checkpoint=torch.load(task/family/str(seed)/'model.pt',weights_only=False,map_location='cpu');assert checkpoint['train_donor_mapping']==mapping
                for k,v in f['transform'].items():np.testing.assert_allclose(v,checkpoint['feature_transform'][k],rtol=1e-5,atol=1e-6)
                model=model_for(family,y.shape[1],len(donors)).cuda();model.load_state_dict(checkpoint['state_dict'])
                cp=prediction(model,f,ci,dc);tp=prediction(model,f,ti,dc)
                np.divide(cp,3.,out=cp);np.divide(tp,3.,out=tp)
                if cpmean is None:cpmean=cp;tpmean=tp
                else:cpmean+=cp;tpmean+=tp
                del model,checkpoint,cp,tp;gc.collect();torch.cuda.empty_cache()
            value=calibration(y,m,ci,cpmean,ti,tpmean,f['std']);value.update({'fold':fold,'family':family,'prediction':'three-seed ensemble, FP32 accumulation','test_donor':str(m.loc[m.partition.eq('test'),'donor_id'].iloc[0]),'test_mae_of_ensemble':selection_mae(y,m,ti,tpmean),'interval_is_not_a_Bayesian_credible_interval':True})
            write_json(p,value);del cpmean,tpmean;gc.collect()
        del f;gc.collect()
    write_json(ROOT/'calibration_status.json',{'state':'complete','cases':16,'updated_epoch':time.time()})
if __name__=='__main__':
    lock=ROOT/'calibration.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:main()
    finally:lock.unlink(missing_ok=True)
