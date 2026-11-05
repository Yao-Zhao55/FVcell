import unittest,tempfile
import completion_core as c
from test_full_suite import FullSuiteTests

class CompletionTests(unittest.TestCase):
    def test_fast_scores_and_intervals_equal_original_definitions(self):
        from completion_metrics import fast_scores,fast_calibration
        from full_suite_core import scores
        from full_suite_run import calibration
        rng=c.np.random.default_rng(44);y=rng.normal(size=(90,21)).astype(c.np.float32);m=c.pd.DataFrame({'index':[str(i) for i in range(90)],'target_contrast':[str(i//3) for i in range(90)],'culture_condition':['Rest','Stim8hr','Stim48hr']*30,'donor_id':['d1']*45+['d2']*45});ci=c.np.arange(30);ti=c.np.arange(30,90);cp=rng.normal(size=(30,21)).astype(c.np.float32);tp=rng.normal(size=(60,21)).astype(c.np.float32);scale=c.np.linspace(.05,2,21,dtype=c.np.float32)
        self.assertEqual(scores(y,m,ti,tp),fast_scores(y,m,ti,tp));self.assertEqual(calibration(y,m,ci,cp,ti,tp,scale),fast_calibration(y,m,ci,cp,ti,tp,scale))
    def test_categorical_reparameterization_preserves_predictor_and_blocks_bypass(self):
        c.torch.manual_seed(1);old=c.Hierarchical(5,2,'hierarchical').eval();u=c.torch.randn(6,65);z=c.torch.randn(6,35);ctx=c.torch.tensor([0,1,2,0,1,2]);d=c.torch.tensor([0,1,0,1,0,1]);z[:,32:]=c.torch.eye(3)[ctx]
        before=old(u,z,ctx,d)[0].detach();f={'z':z.numpy(),'context_codes':ctx.numpy()};m=c.pd.DataFrame({'partition':['train']*6});c.convert_context_ID_path(old,f,m);new=c.IdentifiableHierarchy(5,2,'hierarchical').eval();new.load_state_dict(old.state_dict());after=new(u,z,ctx,d)[0]
        c.torch.testing.assert_close(before,after,atol=1e-7,rtol=1e-5);self.assertTrue(bool((new.state.weight[:,32:]==0).all()))
        zz=z.clone();zz[:,32:]=1e5;c.torch.testing.assert_close(after,new(u,zz,ctx,d)[0])
    def test_artificial_missingness_is_nested_without_using_labels(self):
        a=c.observed_mask_row('sample',1000,.3);b=c.observed_mask_row('sample',1000,.7);self.assertFalse(bool((a&~b).any()))
    def fixture(self):
        y,m,g=FullSuiteTests().fixture();q=c.np.ones(y.shape,bool);q[0,0]=False;q[1,1]=False;q[3,2]=False;q[4,3]=False;q[6,4]=False
        return y,m,g,q
    def test_missing_training_values_and_all_held_labels_do_not_affect_features(self):
        y,m,g,q=self.fixture();a,f=c.mask_features(y,m,g,q);z=y.copy();z[~q]=float('nan');z[~m.partition.eq('train')]=1e5;b,h=c.mask_features(z,m,g,q)
        for k in ('u','z','mean','std'):c.np.testing.assert_allclose(f[k],h[k],atol=1e-7)
        c.np.testing.assert_array_equal(a[c.np.flatnonzero(m.partition.eq('train'))],b[c.np.flatnonzero(m.partition.eq('train'))])
    def test_masked_loss_does_not_use_hidden_labels_or_gradients(self):
        p=c.torch.tensor([[.5,.3]],requires_grad=True);q=c.torch.tensor([[True,False]]);a=c.observed_loss(p,c.torch.tensor([[.2,float('nan')]]),q);a.backward();self.assertEqual(float(p.grad[0,1]),0.)
        b=c.observed_loss(p,c.torch.tensor([[.2,1e8]]),q);self.assertEqual(float(a.detach()),float(b.detach()))
    def test_unseen_donor_has_exactly_zero_random_offset(self):
        c.torch.manual_seed(4);model=c.Hierarchical(5,2,'hierarchical').eval();u=c.torch.randn(3,65);z=c.torch.randn(3,35);ctx=c.torch.arange(3);d=c.torch.full((3,),-1,dtype=c.torch.long);a=model(u,z,ctx,d)[0].detach()
        with c.torch.no_grad():model.donor_offset.normal_(10,4)
        c.torch.testing.assert_close(a,model(u,z,ctx,d)[0]);model.prior().backward();self.assertTrue(bool(c.torch.isfinite(model.log_context.grad).all()))
    def test_P0_has_no_control_expression_features(self):
        y,m,g,q=self.fixture();f=c.context_features(y,m,g,'P0');c.np.testing.assert_array_equal(f['z'][:,:32],0.)
    def test_synthetic_GPU_training_smoke_only(self):
        y,m,g,q=self.fixture();proxy,f=c.mask_features(y,m,g,q);old=dict(c.PLAN)
        try:
            c.PLAN.update({'steps':2,'validate_every':1})
            with tempfile.TemporaryDirectory(dir=c.ROOT) as d:
                model,info=c.fit_new(proxy,m,f,'hierarchical',11,.001,.0003,c.Path(d),c.np.zeros(len(m),c.np.int64),1,q);self.assertEqual(info['steps'],2);self.assertTrue(c.np.isfinite(info['best_selection_mae']))
        finally:c.PLAN.clear();c.PLAN.update(old)
if __name__=='__main__':unittest.main()
