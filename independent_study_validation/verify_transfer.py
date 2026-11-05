"""Verify complete committed ranges on disk; this is not a publisher checksum."""
from pathlib import Path
import json,hashlib,time,math,os
from download_counts import CHUNK
R=Path(__file__).resolve().parent
state=json.loads((R/'download_status.json').read_text())
assert state['state'] in ['download_complete_pending_gzip_crc','complete']
size=state['expected_bytes'];chunk=state.get('chunk_bytes',CHUNK);records=state['chunks'];n=math.ceil(size/chunk)
assert set(records)==set(map(str,range(n)))
assert sum(v['bytes'] for v in records.values())==size
p=R/'data/GSE278572_matrix.mtx.gz';assert p.stat().st_size==size
h=hashlib.sha256()
with p.open('rb') as f:
 for i in range(n):
  wanted=min(chunk,size-i*chunk);b=f.read(wanted);assert len(b)==wanted
  assert records[str(i)]['bytes']==wanted
  assert hashlib.sha256(b).hexdigest()==records[str(i)]['sha256'],f'Committed range {i} differs from its received digest'
  h.update(b)
 assert not f.read(1)
assert h.hexdigest()==state['sha256']
result={'state':'all_received_ranges_verified_on_disk','bytes':size,'ranges':n,'sha256':h.hexdigest(),'received_range_hashes_match_disk':True,'publisher_checksum_available':False,'gzip_CRC':'pending preprocessing stream; see count_provenance.json for completion','biological_validation_complete':False,'updated_epoch':time.time()}
out=R/'transfer_verification.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
