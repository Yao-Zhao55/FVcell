"""Synthetic integration checks only; not biological training results."""
import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import run_first_training as m

class TrainingTests(unittest.TestCase):
    def test_shared_input_excludes_own_condition(self):
        y=np.array([[1.,np.nan],[5.,2.],[100.,999.]],np.float32)
        code=np.array([0,0,0]);train=np.array([True,True,False]);g=np.array([3.,2.],np.float32)
        sums,counts=m.fit_shared(y,code,train)
        np.testing.assert_allclose(m.shared_row(y,0,code,train,sums,counts,g,True),[5,2])
        np.testing.assert_allclose(m.shared_row(y,2,code,train,sums,counts,g),[3,2])
        y[2]=[-9999,-9999]
        a,b=m.fit_shared(y,code,train)
        np.testing.assert_array_equal(a,sums);np.testing.assert_array_equal(b,counts)

    def test_nonfinite_prediction_cannot_remove_gold(self):
        with self.assertRaises(ValueError):m.metrics(np.array([[1.,2.]]),[0],lambda i:np.array([1.,np.nan]),np.array(['Rest']),'test')

    def test_csr_retains_zero_and_missing(self):
        import io
        with m.h5py.File(io.BytesIO(),'w') as f:
            g=f.create_group('matrix');g.attrs['encoding-type']='csr_matrix';g.attrs['shape']=(2,3)
            g.create_dataset('indptr',data=[0,2,3]);g.create_dataset('indices',data=[0,2,1]);g.create_dataset('data',data=[1.,np.nan,4.])
            actual=m.rows(g,0,2)
            np.testing.assert_allclose(actual,np.array([[1,0,np.nan],[0,4,0]]),equal_nan=True)
            cat=f.create_group('category');cat.create_dataset('categories',data=np.array(['a','b'],dtype=m.h5py.string_dtype()));cat.create_dataset('codes',data=[0,-1,1])
            self.assertEqual(m.column(cat).tolist(),['a',None,'b'])

    def test_end_to_end_synthetic_only(self):
        with tempfile.TemporaryDirectory(dir=m.BASE) as temp:
            base=Path(temp);(base/'large_data').mkdir();(base/'prepared').mkdir();out=base/'training_first';out.mkdir()
            old_base,old_out=m.BASE,m.OUT;m.BASE,m.OUT=base,out
            try:
                ids=[];targets=[];ctx=[];partition=[];values=[]
                rng=np.random.default_rng(1)
                for t in range(12):
                    v=rng.normal(size=5)
                    for c in range(3):
                        ids.append(f'T{t}_C{c}');targets.append(f'T{t}');ctx.append(f'C{c}')
                        partition.append('validation' if t<3 and c==0 else 'test' if 3<=t<6 and c==1 else 'train')
                        values.append(v+c*.2)
                splits=pd.DataFrame({'index':ids,'target_contrast':targets,'culture_condition':ctx,'combo_partition':partition})
                splits.to_csv(base/'prepared/condition_and_global_target_splits.csv',index=False)
                freeze={'sha256':{'condition_and_global_target_splits.csv':m.sha(base/'prepared/condition_and_global_target_splits.csv')}}
                (base/'prepared/FREEZE_LOCK.json').write_text(json.dumps(freeze))
                def strings(group,key,val):group.create_dataset(key,data=np.array(val,dtype=m.h5py.string_dtype()))
                p=base/'large_data/de.h5ad'
                with m.h5py.File(p,'w') as f:
                    obs=f.create_group('obs');obs.attrs['_index']='_index'
                    strings(obs,'_index',ids);strings(obs,'target_contrast',targets);strings(obs,'culture_condition',ctx)
                    var=f.create_group('var');var.attrs['_index']='_index';strings(var,'_index',[f'G{i}' for i in range(5)])
                    y=np.array(values,np.float32);y[0,0]=np.nan;y[2,1]=0
                    f.create_group('layers').create_dataset('log_fc',data=y)
                q=base/'large_data/pb.h5ad'
                with m.h5py.File(q,'w') as f:
                    obs=f.create_group('obs');strings(obs,'perturbed_gene_id',['NTC']*3);strings(obs,'guide_type',['non-targeting']*3);obs.create_dataset('keep_for_DE',data=np.ones(3,dtype=bool));strings(obs,'culture_condition',['C0','C1','C2']);strings(obs,'donor_id',['D1']*3)
                    f.create_dataset('X',data=np.ones((3,5),np.int32))
                manifest={}
                for key,path in [('de_stats',p),('pseudobulk',q)]:
                    manifest[key]={'key':path.name,'bytes':path.stat().st_size,'etag':'fixture-only'}
                    path.with_name(path.name+'.verified.json').write_text(json.dumps({'etag':'fixture-only','sha256':m.sha(path)}))
                (base/'large_asset_manifest.json').write_text(json.dumps(manifest))
                m.main()
                result=json.loads((out/'results.json').read_text())
                self.assertTrue(result['training_only_fit'])
                self.assertEqual(len(result['test_results']),5)
                self.assertEqual(result['test_results']['zero_response']['rows'],3)
                self.assertTrue((out/'input_validation.json').exists())
                self.assertFalse((out/'training.lock').exists())
            finally:m.BASE,m.OUT=old_base,old_out

if __name__=='__main__':unittest.main()
