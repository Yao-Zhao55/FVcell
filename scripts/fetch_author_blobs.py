"""Download text blobs by immutable Git object ID; no upstream code execution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,shutil,base64,hashlib
BASE=Path(__file__).resolve().parents[1]
manifest=json.loads((BASE/'cloud_download_manifest.json').read_text(encoding='utf-8'))
items=[x for x in manifest['files'] if x['role']=='author_control_and_unit_definition']
def fetch(item):
    metadata=BASE/'sources'/(item['name']+'.github_blob.json')
    url='https://api.github.com/repos/emdann/GWT_perturbseq_analysis_2025/git/blobs/'+item['git_blob_sha1']
    result=subprocess.run([shutil.which('curl') or 'curl','-L','--fail','--silent','--show-error','--connect-timeout','10','--max-time','30',url,'-o',str(metadata)],capture_output=True,text=True)
    if result.returncode:return {'name':item['name'],'status':'failed','error':result.stderr}
    obj=json.loads(metadata.read_text(encoding='utf-8'))
    data=base64.b64decode(obj['content'])
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if actual!=item['git_blob_sha1']:raise RuntimeError('Blob checksum mismatch')
    (BASE/'sources'/item['name']).write_bytes(data)
    return {'name':item['name'],'status':'verified','git_blob_sha1':actual,'bytes':len(data),'url':url}
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,items))
(BASE/'author_definition_checksums.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False),flush=True)
