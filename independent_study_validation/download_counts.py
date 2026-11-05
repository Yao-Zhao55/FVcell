from pathlib import Path
import os,json,time,urllib.request,concurrent.futures,hashlib,threading,traceback
R=Path(__file__).resolve().parent;D=R/'data';D.mkdir(exist_ok=True)
URL='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl/GSE278572_matrix.mtx.gz'
SIZE=3857480145;CHUNK=2*1024*1024;P=D/'GSE278572_matrix.mtx.gz';STATE=R/'download_status.json';LOCK=R/'download.lock'
M=threading.Lock();records={}
def write_state(state,**kw):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps({'state':state,'pid':os.getpid(),'url':URL,'expected_bytes':SIZE,'verified_bytes':sum(v['bytes'] for v in records.values()),'chunks':records,'updated_epoch':time.time(),**kw},indent=2),encoding='utf-8');os.replace(p,STATE)
def chunk(i):
 a=i*CHUNK;b=min(SIZE,a+CHUNK)-1
 if str(i) in records:return
 for attempt in range(8):
  try:
   o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
   req=urllib.request.Request(URL,headers={'Range':f'bytes={a}-{b}','User-Agent':'Mozilla/5.0'})
   h=hashlib.sha256();n=0
   with o.open(req,timeout=90) as f,open(P,'r+b',buffering=0) as out:
    if f.status!=206 or f.headers.get('Content-Range')!=f'bytes {a}-{b}/{SIZE}':raise ValueError('Invalid range response')
    out.seek(a)
    while True:
     v=f.read(1024*1024)
     if not v:break
     if n+len(v)>b-a+1:raise ValueError('Range overflow')
     out.write(v);h.update(v);n+=len(v)
   if n!=b-a+1:raise ValueError(f'Truncated chunk {n}')
   with M:records[str(i)]={'bytes':n,'sha256':h.hexdigest()};write_state('downloading',chunk_bytes=CHUNK)
   return
  except Exception:
   if attempt==7:raise
   time.sleep(min(30,2**attempt))
if __name__=='__main__':
 fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  if STATE.exists():
   previous=json.loads(STATE.read_text());records=previous.get('chunks',{})
   if records and previous.get('chunk_bytes',32*1024*1024)!=CHUNK:raise ValueError('Chunk mapping changed with committed ranges')
  if not P.exists():
   with open(P,'wb') as f:f.truncate(SIZE)
  if P.stat().st_size!=SIZE:raise ValueError('Unexpected file size')
  write_state('downloading')
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(chunk,range((SIZE+CHUNK-1)//CHUNK)))
  h=hashlib.sha256()
  with open(P,'rb') as f:
   while v:=f.read(8*1024*1024):h.update(v)
  write_state('download_complete_pending_gzip_crc',sha256=h.hexdigest(),publisher_checksum_available=False)
 except BaseException as e:write_state('failed',error=str(e),traceback=traceback.format_exc());raise
 finally:LOCK.unlink(missing_ok=True)
