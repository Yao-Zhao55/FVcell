import unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from atomic_json import write_json
class StateWriteTests(unittest.TestCase):
    def test_transient_windows_lock_retries(self):
        p=MagicMock();temp=MagicMock();p.suffix='.json';p.with_suffix.return_value=temp
        temp.replace.side_effect=[PermissionError('sharing violation'),PermissionError('sharing violation'),None]
        with patch('atomic_json.time.sleep') as sleep:
            write_json(p,{'state':'test'})
        self.assertEqual(temp.replace.call_count,3)
        self.assertEqual(sleep.call_count,2)
    def test_permanent_error_is_bounded(self):
        p=MagicMock();temp=MagicMock();p.suffix='.json';p.with_suffix.return_value=temp;temp.replace.side_effect=PermissionError('denied')
        with patch('atomic_json.time.sleep'),self.assertRaises(PermissionError):write_json(p,{})
        self.assertEqual(temp.replace.call_count,50)
if __name__=='__main__':unittest.main()
