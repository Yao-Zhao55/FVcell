from pathlib import Path
import urllib.request,concurrent.futures,time,json
R=Path(__file__).resolve().parent
URLS=[
'https://zenodo.org/records/13924126/files/perturbseq_analysis_pipeline.zip?download=1',
'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl/GSE278572_matrix.mtx.gz?download=1',
'http://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl/GSE278572_matrix.mtx.gz',
]
def probe(u):
 t=time.time();o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 try:
  with o.open(urllib.request.Request(u,headers={'Range':'bytes=-100000','User-Agent':'Mozilla/5.0'}),timeout=20) as f:
   b=f.read(100000);headers=dict(f.headers);status=f.status
  name='route_'+str(URLS.index(u))+'.bin';(R/'sources'/name).write_bytes(b)
  return {'url':u,'seconds':time.time()-t,'status':status,'bytes':len(b),'headers':headers,'saved':name}
 except Exception as e:return {'url':u,'seconds':time.time()-t,'error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  a=list(ex.map(probe,URLS));print(json.dumps(a,indent=2));(R/'sources/route_probe.json').write_text(json.dumps(a,indent=2))
