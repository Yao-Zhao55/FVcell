from pathlib import Path
import urllib.request,json,concurrent.futures,gzip
R=Path(__file__).resolve().parent;S=R/'sources'
O=urllib.request.build_opener(urllib.request.ProxyHandler({}))
URLS={
'arce_zenodo.json':'https://zenodo.org/api/records/13924126',
'gsm_example.txt':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8549475&targ=self&form=text&view=full',
'GSE278572_barcodes.tsv.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl/GSE278572_barcodes.tsv.gz',
'GSE278572_features.tsv.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl/GSE278572_features.tsv.gz',
'GSE278572_protospacer_calls_per_cell.csv.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl/GSE278572_protospacer_calls_per_cell.csv.gz',
}
def get(item):
 n,u=item
 try:
  p=S/n
  if not p.exists():
   with O.open(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=50) as f:p.write_bytes(f.read(15000000))
  b=p.read_bytes()
  if n.endswith('.json'):
   j=json.loads(b);out=json.dumps({'description':j.get('metadata',{}).get('description'),'files':j.get('files')},ensure_ascii=False)[:14000]
  elif n.endswith('.gz'):out=gzip.decompress(b).decode()[:4500]
  else:out=b.decode()[:12500]
  return n,out
 except Exception as e:return n,str(e)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
  for n,v in ex.map(get,URLS.items()):print('\nFILE',n,'\n',v,flush=True)
