"""Portable, resumable downloads; incomplete files never masquerade as inputs."""
from pathlib import Path
import json,hashlib,subprocess,shutil,argparse,sys,os,time
BASE=Path(__file__).resolve().parents[1]
def check(path,item):
    if not path.is_file() or path.stat().st_size!=item['bytes']:return False
    kind='sha1' if item.get('git_blob_sha1') else 'md5' if item.get('md5') else 'sha256'
    digest=hashlib.new(kind)
    if item.get('git_blob_sha1'):
        digest.update(b'blob '+str(item['bytes']).encode()+b'\0')
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    expected=item.get('git_blob_sha1') or item.get('md5') or item.get('sha256')
    return digest.hexdigest()==expected
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only',nargs='*')
    parser.add_argument('--allow-large',action='store_true')
    parser.add_argument('--timeout',type=int,default=300)
    args=parser.parse_args()
    manifest=json.loads((BASE/'cloud_download_manifest.json').read_text(encoding='utf-8'))
    results=[]
    for item in manifest['files']:
        if args.only and item['name'] not in args.only:continue
        if item['bytes']>134217728 and not args.allow_large:
            results.append({'name':item['name'],'status':'not_requested_large_file'});continue
        target=BASE/'sources'/item['name'];target.parent.mkdir(exist_ok=True)
        partial=target.with_name(target.name+'.download')
        if check(target,item):
            results.append({'name':item['name'],'status':'verified_cached'});continue
        if target.exists():
            if partial.exists():raise RuntimeError('Both invalid target and partial exist; inspect manually')
            target.replace(partial)
        code=1
        for attempt in range(2):
            result=subprocess.run([shutil.which('curl') or 'curl','-L','--fail','--silent','--show-error','--continue-at','-','--connect-timeout','15','--max-time',str(args.timeout),item['url'],'-o',str(partial)],capture_output=True,text=True,encoding='utf-8',errors='replace')
            code=result.returncode
            if code==0:break
            print(json.dumps({'name':item['name'],'attempt':attempt+1,'error':result.stderr[-500:]},ensure_ascii=False),flush=True)
        if check(partial,item):
            partial.replace(target);status='verified'
        else:status='incomplete_or_checksum_failed'
        results.append({'name':item['name'],'status':status,'returncode':code,'bytes_downloaded':target.stat().st_size if status=='verified' else partial.stat().st_size if partial.exists() else 0})
        (BASE/'verified_download_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    if any(r['status']=='incomplete_or_checksum_failed' for r in results):sys.exit(1)
if __name__=='__main__':main()
