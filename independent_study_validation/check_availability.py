from pathlib import Path
import urllib.request,json,xml.etree.ElementTree as ET,concurrent.futures
R=Path(__file__).resolve().parent; S=R/'sources'
URLS={
'author_repository_tree.json':'https://api.github.com/repos/anniqueclaringbould/CD4_variant2CRE2gene2network/git/trees/main?recursive=1',
'zenodo_metadata.json':'https://zenodo.org/api/records/18876252',
'GSE278572_directory.html':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl/',
'author_readme.md':'https://raw.githubusercontent.com/anniqueclaringbould/CD4_variant2CRE2gene2network/main/README.md',
}
def get(item):
 n,u=item
 try:
  o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with o.open(urllib.request.Request(u,headers={'User-Agent':'Research-data-availability-audit'}),timeout=45) as f: b=f.read(5000000)
  (S/n).write_bytes(b)
  if 'tree' in n:
   j=json.loads(b); out='\n'.join(a['path'] for a in j.get('tree',[]) if any(k in a['path'].lower() for k in ['readme','data','csv','tsv','rds','h5','pseudobulk','deseq']))
  elif 'zenodo' in n:
   j=json.loads(b);out=json.dumps({k:j.get(k) for k in ['access','files','links']})[:9000]
  else:out=b.decode('utf-8','replace')[:14000]
  return n,out
 except Exception as e:return n,str(e)
if __name__=='__main__':
 x=ET.parse(S/'independent_paper.xml')
 print('\n'.join(' '.join(p.itertext()) for p in x.iter('p') if any(k in ' '.join(p.itertext()).lower() for k in ['zenodo','github','data availability'])))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  for n,v in ex.map(get,URLS.items()):print('\nFILE',n,'\n',v,flush=True)
