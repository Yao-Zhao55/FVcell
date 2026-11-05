"""Software tests only; artificial fixtures are not biological results."""
import unittest
import pandas as pd
from complete_priorities import validate_permissions,equal_donor_mae
class Guards(unittest.TestCase):
    def test_global_unseen_cannot_read_same_target(self):
        with self.assertRaises(ValueError):validate_permissions('P2',True,same_target_response=True)
    def test_held_condition_never_allowed(self):
        for p in ['P0','P1','P2']:
            with self.assertRaises(ValueError):validate_permissions(p,False,target_condition_response=True)
    def test_strict_P1_forbids_held_target_evidence(self):
        with self.assertRaises(ValueError):validate_permissions('P1',False,same_target_response=True)
    def test_allowed_P2_support(self):
        self.assertTrue(validate_permissions('P2',False,same_target_response=True))
    def test_donors_not_weighted_by_target_count(self):
        f=pd.DataFrame([('a','c',str(i),0.) for i in range(100)]+[('b','c','x',2.)],columns=['donor','context','target','mae'])
        self.assertAlmostEqual(equal_donor_mae(f),1.)
    def test_nonfinite_is_not_zero_imputed(self):
        f=pd.DataFrame([('a','c','x',float('nan'))],columns=['donor','context','target','mae'])
        with self.assertRaises(ValueError):equal_donor_mae(f)
if __name__=='__main__':unittest.main()
