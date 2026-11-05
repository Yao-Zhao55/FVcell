"""Complete the small condition index using bounded S3 range requests."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,shutil,hashlib,json,re
BASE=Path(__file__).resolve().parents[1]
item=next(x for x in json.loads((BASE/'cloud_download_manifest.json').read_text(encoding='utf-8'))['files'] if x['name']=='GWT_DE_stats.suppl_table.csv')
target=BASE/'sources'/item['name']
partial=target.with_name(target.name+'.download')
out=BASE/'sources'/'index_chunks';out.mkdir(exist_ok=True)
prefix=partial.read_bytes() if partial.exists() else b''
URL='https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/suppl_tables/DE_stats.suppl_table.csv'
def fetch(pair):
    start,end=pair
    p=out/f'{start}-{end}.part';h=out/f'{start}-{end}.headers'
    if not p.exists() or p.stat().st_size!=end-start+1:
        result=subprocess.run([shutil.which('curl') or 'curl','-L','--fail','--silent','--show-error','--connect-timeout','10','--max-time','35','--retry','1','--max-filesize',str(end-start+1),'--range',f'{start}-{end}',URL,'-D',str(h),'-o',str(p)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
    if p.stat().st_size!=end-start+1 or f'bytes {start}-{end}/' not in h.read_text(errors='replace'):
        raise RuntimeError('Unexpected range response')
    return p
ranges=[(i,min(i+131072-1,item['bytes']-1)) for i in range(len(prefix),item['bytes'],131072)]
with ThreadPoolExecutor(max_workers=4) as pool:
    parts=list(pool.map(fetch,ranges))
data=prefix+b''.join(p.read_bytes() for p in parts)
actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
if actual!=item['git_blob_sha1'] or len(data)!=item['bytes']:raise RuntimeError('Author repository checksum mismatch')
target.write_bytes(data)
report={'file':item['name'],'bytes':len(data),'git_blob_sha1':actual,'matches_pinned_repository':True,'source_url':URL,'requests':len(ranges),'sha256':hashlib.sha256(data).hexdigest()}
(BASE/'condition_index_checksum.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
