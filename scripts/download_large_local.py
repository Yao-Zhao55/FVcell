"""Bounded, resumable local downloads of the two approved processed datasets.

Part files are preallocated; progress is measured from verified chunks, NOT size.
Source identity is pinned with S3 If-Match/ETag; multipart ETags are not MD5s.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json,hashlib,shutil,subprocess,re,time,os,argparse
from atomic_json import write_json
BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'large_data'
CHUNK=2*1024*1024
BUDGET=100_000_000_000
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save_json(path,obj):
    write_json(path,obj)
def download(asset_name,asset,workers):
    target=DATA/Path(asset['key']).name
    partial=target.with_name(target.name+'.download')
    checkpoint=target.with_name(target.name+'.chunks.json')
    receipt=target.with_name(target.name+'.verified.json')
    if target.exists():
        if not receipt.exists():raise RuntimeError('Existing final file without a receipt; do not overwrite')
        record=json.loads(receipt.read_text(encoding='utf-8'))
        if target.stat().st_size!=asset['bytes'] or record['etag']!=asset['etag']:raise RuntimeError('Final file does not match manifest')
        return record
    state=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {'etag':asset['etag'],'bytes':asset['bytes'],'chunk_size':CHUNK,'chunks':{}}
    if state['etag']!=asset['etag'] or state['bytes']!=asset['bytes'] or state['chunk_size']!=CHUNK:raise RuntimeError('Incompatible checkpoint')
    if not partial.exists():
        if state['chunks']:raise RuntimeError('Checkpoint exists but data missing')
        with partial.open('wb') as f:f.truncate(asset['bytes'])
    elif partial.stat().st_size!=asset['bytes']:raise RuntimeError('Unexpected partial size')
    # Recheck existing completed chunks before resuming.
    with partial.open('rb') as f:
        for start,info in state['chunks'].items():
            f.seek(int(start))
            if hashlib.sha256(f.read(info['bytes'])).hexdigest()!=info['sha256']:raise RuntimeError('Stored chunk checksum mismatch')
    pieces=[(i,min(i+CHUNK-1,asset['bytes']-1)) for i in range(0,asset['bytes'],CHUNK) if str(i) not in state['chunks']]
    tempdir=DATA/'transfer_chunks';tempdir.mkdir(exist_ok=True)
    def fetch(pair):
        start,end=pair
        p=tempdir/f'{asset_name}_{start}.chunk';hdr=tempdir/f'{asset_name}_{start}.headers'
        def valid():
            header=hdr.read_text(errors='replace') if hdr.exists() else ''
            match=re.findall(r'(?im)^etag:\s*(.+?)\s*$',header)
            return p.exists() and p.stat().st_size==end-start+1 and f'bytes {start}-{end}/{asset["bytes"]}' in header and bool(match) and match[-1].strip()==asset['etag']
        # A stopped worker can leave finished, identity-checked ranges on disk.
        if valid():return start,p,end-start+1,hdr
        messages=[]
        for attempt in range(3):
            cmd=[shutil.which('curl') or 'curl','-L','--fail','--silent','--show-error','--connect-timeout','15','--max-time','600','--max-filesize',str(end-start+1),'--header','If-Match: '+asset['etag'],'--range',f'{start}-{end}',asset['url'],'-D',str(hdr),'-o',str(p)]
            r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
            if r.returncode==0 and valid():
                return start,p,end-start+1,hdr
            messages.append(r.stderr[-300:] or 'Range/ETag validation failed')
            time.sleep(min(5,attempt+1))
        raise RuntimeError(str(start)+' '+repr(messages))
    status_path=BASE/'download_status.json'
    start_time=time.time()
    completed_initial=sum(x['bytes'] for x in state['chunks'].values())
    def status(error=None):
        done=sum(x['bytes'] for x in state['chunks'].values())
        elapsed=time.time()-start_time
        obj={'asset':asset_name,'file':str(target),'expected_bytes':asset['bytes'],'verified_downloaded_bytes':done,'percent':round(100*done/asset['bytes'],4),'workers':workers,'current_session_bytes_per_second':round((done-completed_initial)/max(elapsed,1),1),'state':'paused_after_error' if error else 'downloading','error':error,'partial_file_is_preallocated_not_complete':True,'updated_epoch':time.time(),'remaining_asset':'pseudobulk' if asset_name=='de_stats' else None}
        save_json(status_path,obj)
        return obj
    status()
    # Limit queued work so cancellation/error cannot leave thousands of requests.
    # The slow overseas link needs up to ten minutes per 2 MiB range.
    todo=iter(pieces)
    with ThreadPoolExecutor(max_workers=workers) as pool,partial.open('r+b') as f:
        pending={}
        for _ in range(workers):
            pair=next(todo,None)
            if pair is not None:pending[pool.submit(fetch,pair)]=pair
        try:
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    start,p,size,hdr=future.result()
                    data=p.read_bytes();f.seek(start);f.write(data);f.flush()
                    state['chunks'][str(start)]={'bytes':size,'sha256':hashlib.sha256(data).hexdigest()}
                    save_json(checkpoint,state)
                    p.unlink();hdr.unlink()
                    current=status()
                    print(json.dumps(current,ensure_ascii=False),flush=True)
                    pair=next(todo,None)
                    if pair is not None:pending[pool.submit(fetch,pair)]=pair
        except BaseException as error:
            for future in pending:future.cancel()
            status(str(error));raise
    if sum(x['bytes'] for x in state['chunks'].values())!=asset['bytes']:raise RuntimeError('Incomplete')
    with partial.open('rb') as f:
        if f.read(8)!=b'\x89HDF\r\n\x1a\n':raise RuntimeError('Expected HDF5 file signature')
    record={'asset':asset_name,'bytes':asset['bytes'],'etag':asset['etag'],'sha256':digest(partial),'validation':'pinned S3 object ETag on every range, per-chunk SHA256, complete size and HDF5 signature; source multipart ETag is not a published MD5'}
    partial.replace(target);save_json(receipt,record)
    return record
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if not 1<=args.workers<=64:raise SystemExit('Use 1-64 workers')
    DATA.mkdir(exist_ok=True)
    lock=BASE/'download_worker.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit('Worker lock exists; check prior process before restarting')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        manifest=json.loads((BASE/'large_asset_manifest.json').read_text(encoding='utf-8'))
        approved=['de_stats','pseudobulk']
        total=sum(manifest[k]['bytes'] for k in approved)
        if total>BUDGET:raise RuntimeError('100 GB approval limit exceeded')
        required=sum(manifest[k]['bytes'] for k in approved if not (DATA/Path(manifest[k]['key']).name).exists() and not (DATA/(Path(manifest[k]['key']).name+'.download')).exists())
        if shutil.disk_usage(DATA).free<required+10_000_000_000:raise RuntimeError('Insufficient disk headroom')
        receipts=[download(k,manifest[k],args.workers) for k in approved]
        save_json(BASE/'download_status.json',{'state':'complete','files':receipts,'total_bytes':total,'updated_epoch':time.time()})
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
