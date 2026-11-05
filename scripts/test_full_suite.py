import unittest
import full_suite_core as c
import numpy as np,pandas as pd
class FullSuiteTests(unittest.TestCase):
    def fixture(self):
        targets=np.load(c.PREP/'mechanism.npz')['target_ids'][:3];genes=pd.read_csv(c.PREP/'primary_genes.csv').gene_id.to_numpy()[:5]
        rows=[]
        partitions=['train','train','test','train','train','test','train','validation','validation']
        for i in range(9):rows.append({'index':str(i),'target_contrast':targets[i//3],'culture_condition':['Rest','Stim48hr','Stim8hr'][i%3],'partition':partitions[i],'selection_or_calibration':'calibration' if i==8 else 'selection'})
        return np.random.default_rng(1).normal(size=(9,5)).astype(np.float32),pd.DataFrame(rows),genes
    def test_held_response_cannot_change_features_or_transform(self):
        y,m,g=self.fixture();a=c.features(y,m,g,'P2');z=y.copy();z[~m.partition.eq('train')]=10000
        b=c.features(z,m,g,'P2')
        for key in ('u','z','mean','std'):np.testing.assert_allclose(a[key],b[key])
    def test_P1_has_no_same_target_effect_features(self):
        y,m,g=self.fixture();a=c.features(y,m,g,'P1');z=y*99;b=c.features(z,m,g,'P1')
        np.testing.assert_allclose(a['u'],b['u']);np.testing.assert_allclose(a['z'],b['z']);np.testing.assert_array_equal(a['u'][:,32:],0)
    def test_strict_external_evidence_changes_only_P2_inputs(self):
        y,m,g=self.fixture();m['partition']=['train','train','train','evidence_only','evidence_only','test','evidence_only','validation','validation']
        ev=np.array([1,1,1,1,1,0,1,0,0],dtype=bool);m.attrs['evidence_mask']=ev
        a=c.features(y,m,g,'P2');p1=c.features(y,m,g,'P1');z=y.copy();z[3:5]+=100
        b=c.features(z,m,g,'P2');p1b=c.features(z,m,g,'P1')
        np.testing.assert_allclose(a['mean'],b['mean']);np.testing.assert_allclose(p1['u'],p1b['u'])
        self.assertFalse(np.allclose(a['u'][5],b['u'][5]))
        bad=m.copy();bad.attrs['evidence_mask']=ev.copy();bad.attrs['evidence_mask'][5]=True
        with self.assertRaises(ValueError):c.features(y,bad,g,'P2')
    def test_capacity_matching_and_signed_outputs(self):
        models={k:c.build_model(k,10282) for k in ['fv','concat','shared','independent']};n={k:sum(p.numel() for p in v.parameters()) for k,v in models.items()}
        self.assertLess(max(n.values())/min(n.values()),1.02)
        x=c.torch.randn(6,65);z=c.torch.randn(6,35);ctx=c.torch.tensor([0,1,2,0,1,2])
        for k in models:
            model=c.build_model(k,7);pred=model(x,z,ctx)[0];self.assertEqual(tuple(pred.shape),(6,7));self.assertTrue(c.torch.isfinite(pred).all())
    def test_tiny_GPU_training_selection_only(self):
        y,m,g=self.fixture();f=c.features(y,m,g,'P2');folder=c.OUT/'test_artifacts';folder.mkdir(exist_ok=True)
        protocol={'max_steps':4,'batch_size':3,'validation_interval':2,'patience':2,'residual_penalty':.001,'mechanism_penalty':.001}
        model,info=c.fit(y,m,f,'fv',11,.001,folder,protocol)
        self.assertEqual(info['steps'],4);self.assertEqual(len(info['history']),2)
        p=c.predict(model,f,np.array([2,5]),f['mean']);self.assertTrue(np.isfinite(p).all());self.assertEqual(p.shape,(2,5))
    def test_fast_selection_matches_full_metric(self):
        y,m,g=self.fixture();ii=np.array([2,5,7]);pred=y[ii]*.2
        self.assertAlmostEqual(c.selection_mae(y,m,ii,pred),c.scores(y,m,ii,pred)['mae'],places=14)
if __name__=='__main__':unittest.main()
