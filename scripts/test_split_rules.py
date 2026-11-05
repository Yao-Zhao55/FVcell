import unittest,math
from split_rules import combo_split,target_split,observed,control_key,validate_effect_rows,CONDITIONS
class SplitRules(unittest.TestCase):
    def test_order_invariance(self):
        keys=[('ENSG'+str(i),c) for i in range(100) for c in CONDITIONS]
        self.assertEqual({k:combo_split(*k) for k in keys},{k:combo_split(*k) for k in reversed(keys)})
    def test_combo_heldout_has_training_context(self):
        labels=set()
        for i in range(1000):
            values=[combo_split('gene'+str(i),c) for c in CONDITIONS]
            labels.update(values)
            self.assertGreaterEqual(values.count('train'),2)
            self.assertLessEqual(sum(v!='train' for v in values),1)
        self.assertEqual(labels,{'train','validation','test'})
    def test_global_targets_do_not_cross_partitions(self):
        partitions={s:set() for s in ['train','validation','test']}
        for i in range(1000):
            for c in CONDITIONS:partitions[target_split(str(i))].add(str(i))
        self.assertFalse(partitions['train']&partitions['test'])
        self.assertFalse(partitions['validation']&partitions['test'])
    def test_zero_and_nonsignificant_are_observed(self):
        self.assertTrue(observed(0.0));self.assertTrue(observed(0.0001))
        self.assertFalse(observed(float('nan')));self.assertFalse(observed(float('inf')))
    def row(self):return dict(study_id='S',donor_id='D',context_id='C',target_id='T',response_gene_id='G',effect=0.0,observed=True,source_kind='measured_effect')
    def test_duplicate_provenance_fails(self):
        with self.assertRaises(ValueError):validate_effect_rows([self.row(),self.row()])
    def test_teacher_output_is_not_gold(self):
        row=self.row();row['source_kind']='teacher_prediction'
        with self.assertRaises(ValueError):validate_effect_rows([row])
    def test_missing_mask_is_not_zero(self):
        row=self.row();row['effect']=float('nan')
        with self.assertRaises(ValueError):validate_effect_rows([row])
    def test_controls_do_not_cross_donor_or_context(self):
        a=dict(study_id='S',donor_id='D1',context_id='Rest',assay='RNA',readout_time='t')
        b={**a,'donor_id':'D2'};c={**a,'context_id':'Stim'}
        self.assertNotEqual(control_key(a),control_key(b));self.assertNotEqual(control_key(a),control_key(c))
if __name__=='__main__':unittest.main(verbosity=2)
