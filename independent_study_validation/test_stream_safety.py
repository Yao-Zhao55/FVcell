"""Software checks only; generated fixtures are not biological results."""
import unittest,gzip,sys,io
from prepare_external import blocks,np,R
class StreamSafety(unittest.TestCase):
 def fixture(self,raw):
  return io.BytesIO(gzip.compress(raw))
 def test_complete_integer_stream(self):
  p=self.fixture(b'%%MatrixMarket matrix coordinate integer general\n% software fixture\n2 3 3\n1 1 4\n2 2 5\n1 3 2\n');a=list(blocks(p));self.assertEqual(a[0],(2,3,3));np.testing.assert_array_equal(np.concatenate(a[1:]),[[1,1,4],[2,2,5],[1,3,2]])
 def test_reject_incomplete_NNZ(self):
  p=self.fixture(b'%%MatrixMarket matrix coordinate integer general\n2 3 4\n1 1 4\n2 2 5\n1 3 2\n')
  with self.assertRaises(ValueError):list(blocks(p))
 def test_reject_noninteger_schema(self):
  p=self.fixture(b'%%MatrixMarket matrix coordinate real general\n2 3 1\n1 1 0.5\n')
  with self.assertRaises(ValueError):list(blocks(p))
 def test_crc_corruption_detected(self):
  p=self.fixture(b'%%MatrixMarket matrix coordinate integer general\n2 3 1\n1 1 4\n');b=bytearray(p.getvalue());b[-8]^=1;p=io.BytesIO(b)
  with self.assertRaises((OSError,EOFError)):list(blocks(p))
 def test_string_metadata_without_pickle(self):
  a=io.BytesIO();np.savez_compressed(a,gene_ids=np.asarray(['ENSG1','ENSG2'],dtype=str));a.seek(0)
  with np.load(a,allow_pickle=False) as f:self.assertEqual(f['gene_ids'].tolist(),['ENSG1','ENSG2'])
if __name__=='__main__':unittest.main(verbosity=2)
