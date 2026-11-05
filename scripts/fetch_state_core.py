"""Fetch official licensed STATE source by Git blob identity, never pretrained weights."""
from pathlib import Path
import json,base64,subprocess,shutil,hashlib
from concurrent.futures import ThreadPoolExecutor
BASE=Path(__file__).resolve().parents[1]/'full_experiments'
tree=json.loads((BASE/'state_repository_tree.json').read_text(encoding='utf-8'))
paths=['LICENSE','pyproject.toml']+['src/state/tx/models/'+n for n in ['base.py','utils.py','decoders.py','pseudobulk.py','state_transition.py']]
by_path={x['path']:x for x in tree['tree']};dest=BASE/'upstream_state';dest.mkdir(exist_ok=True)
def fetch(path):
    entry=by_path[path];p=dest/path;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.api.json')
    cmd=[shutil.which('curl') or 'curl','-L','--fail','--silent','--show-error','--connect-timeout','15','--max-time','90','https://api.github.com/repos/ArcInstitute/state/git/blobs/'+entry['sha'],'-o',str(tmp)]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode:raise ValueError(r.stderr)
    data=base64.b64decode(json.loads(tmp.read_text())['content']);blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob!=entry['sha']:raise ValueError('Source identity mismatch')
    p.write_bytes(data)
    return {'path':path,'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,paths))
(dest/'provenance.json').write_text(json.dumps({'repository':'https://github.com/ArcInstitute/state','tree_sha':tree['sha'],'files':records,'license':'CC-BY-NC-SA-4.0, original license retained','pretrained_weights_used':False},indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
