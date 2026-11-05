from pathlib import Path
import concurrent.futures,hashlib,json,re,urllib.request,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'sources';SRC.mkdir(parents=True,exist_ok=True)
URLS={
'independent_paper.xml':'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13060838/fullTextXML',
'independent_paper_metadata.json':'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:PMC13060838&format=json',
'GSE278572_latest.txt':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278572&targ=self&form=text&view=full',
}
def get(item):
 name,url=item;p=SRC/name
 if not p.exists():
  o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with o.open(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as r:p.write_bytes(r.read(10_000_000))
 data=p.read_bytes();rec={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 if name.endswith('.xml'):
  root=ET.fromstring(data)
  text='\n'.join(' '.join(el.itertext()) for el in root.iter() if el.tag in ['article-title','p','title','ext-link'])
  (SRC/(name+'.txt')).write_text(text,encoding='utf-8')
  print(text[:4500],flush=True)
  print('\nDATA LINKS',[(e.attrib,''.join(e.itertext())[:100]) for e in root.iter('ext-link') if any(x in ''.join(e.itertext()).lower()+str(e.attrib).lower() for x in ['geo','zenodo','github','figshare','osf','data'])][-35:],flush=True)
 return name,rec
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  results=dict(ex.map(get,URLS.items()))
 (ROOT/'metadata_receipts.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(results,ensure_ascii=False,indent=2))
