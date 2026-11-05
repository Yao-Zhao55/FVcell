from pathlib import Path
import urllib.request,urllib.parse,json,concurrent.futures,time
R=Path(__file__).resolve().parent;D=R/'sources/bibliography';D.mkdir(exist_ok=True)
DOIS={'gwt':'10.1016/j.cell.2026.08.002','replogle':'10.1016/j.cell.2022.05.013','arce':'10.1038/s41586-024-08314-y','state':'10.1016/j.cell.2026.07.052','linear':'10.1038/s41592-025-02772-6','systema':'10.1038/s41587-025-02777-8','calibrated':'10.1038/s41587-026-03307-w','recoverable':'10.64898/2026.09.30.755756','mechanisms':'10.64898/2026.05.08.723625','pie':'10.64898/2026.10.02.756297','scperturb':'10.1038/s41592-023-02144-y'}
def get(item):
 name,doi=item;p=D/(name+'.json')
 try:
  if not p.exists():
   o=urllib.request.build_opener(urllib.request.ProxyHandler({}));u='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
   with o.open(urllib.request.Request(u,headers={'User-Agent':'Academic-manuscript-reference-audit'}),timeout=35) as f:p.write_bytes(f.read(2000000))
  m=json.loads(p.read_text())['message'];return name,{'doi':doi,'title':m.get('title',[''])[0],'authors':[a.get('given','')+' '+a.get('family','') for a in m.get('author',[])],'venue':m.get('container-title',[]),'published':m.get('published'),'type':m.get('type')}
 except Exception as e:return name,{'doi':doi,'error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  a=dict(ex.map(get,DOIS.items()));(R/'bibliography_verified.json').write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(a,ensure_ascii=False,indent=2))
