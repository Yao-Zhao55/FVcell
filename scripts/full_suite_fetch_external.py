"""Bounded public scPerturb downloads, verified against published record MD5."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,shutil,hashlib,re,os,time
from atomic_json import write_json
ROOT=Path(__file__).resolve().parents[1]/'full_experiments';DEST=ROOT/'external_data';DEST.mkdir(exist_ok=True)
record=json.loads((ROOT/'scperturb_record.json').read_text(encoding='utf-8'))
names=['ReplogleWeissman2022_K562_essential.h5ad','ReplogleWeissman2022_rpe1.h5ad']
assets=[f for f in record['files'] if f['key'] in names]
assert len(assets)==2 and sum(f['size'] for f in assets)<3_000_000_000
def md5(p):
    h=hashlib.md5()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
lock=DEST/'worker.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
try:
    for asset in assets:
        p=DEST/asset['key'];receipt=p.with_suffix('.receipt.json');checksum=asset['checksum'].split(':',1)[1]
        if p.exists() and receipt.exists():
            if p.stat().st_size==asset['size'] and md5(p)==checksum:continue
            raise ValueError('Existing external file failed verification')
        parts=DEST/(asset['key']+'.parts');parts.mkdir(exist_ok=True);chunk=2*1024*1024
        spans=[(i,min(i+chunk-1,asset['size']-1)) for i in range(0,asset['size'],chunk)]
        def fetch(pair):
            a,b=pair;q=parts/f'{a}.part';h=parts/f'{a}.headers'
            def valid():return q.exists() and q.stat().st_size==b-a+1 and h.exists() and f'bytes {a}-{b}/{asset["size"]}' in h.read_text(errors='replace')
            if valid():return q
            for attempt in range(4):
                r=subprocess.run([shutil.which('curl') or 'curl','-L','--fail','--silent','--show-error','--connect-timeout','15','--max-time','120','--max-filesize',str(b-a+1),'--range',f'{a}-{b}',asset['links']['self'],'-D',str(h),'-o',str(q)],capture_output=True,text=True)
                if r.returncode==0 and valid():return q
                time.sleep(1+attempt)
            raise ValueError('External range failed: '+str(a)+' '+r.stderr[-200:])
        done=0
        with ThreadPoolExecutor(max_workers=64) as pool:
            for future in as_completed([pool.submit(fetch,s) for s in spans]):
                done+=future.result().stat().st_size
                write_json(DEST/'status.json',{'state':'downloading','file':p.name,'verified_range_bytes':done,'expected_bytes':asset['size'],'updated_epoch':time.time()})
        partial=p.with_suffix(p.suffix+'.download')
        with partial.open('wb') as out:
            for a,b in spans:
                with (parts/f'{a}.part').open('rb') as f:shutil.copyfileobj(f,out)
        if partial.stat().st_size!=asset['size'] or md5(partial)!=checksum:raise ValueError('Published external MD5 mismatch')
        partial.replace(p);write_json(receipt,{'source_record':'https://zenodo.org/records/13350497','bytes':asset['size'],'md5':checksum,'source_url':asset['links']['self']})
        for a,b in spans:(parts/f'{a}.part').unlink();(parts/f'{a}.headers').unlink()
    write_json(DEST/'status.json',{'state':'complete','total_bytes':sum(a['size'] for a in assets),'updated_epoch':time.time()})
except BaseException as e:
    write_json(DEST/'status.json',{'state':'failed','error':str(e),'updated_epoch':time.time()});raise
finally:lock.unlink(missing_ok=True)
