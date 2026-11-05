from pathlib import Path
import json,hashlib,os,ast,gzip,io,sys
from pypdf import PdfReader
import pdfplumber
R=Path(__file__).resolve().parent;B=R.parent;W=B/'conference_manuscript_20261027';PROJECT=B.parents[2]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while v:=f.read(8*1024*1024):h.update(v)
 return h.hexdigest()
manifest=json.loads((B/'protected_files_before.json').read_text(encoding='utf-8'));changed=[]
for item in manifest:
 p=PROJECT/item['path']
 if not p.exists() or sha(p)!=item['sha256']:changed.append(item['path'])
assert not changed,changed
checks=[]
for path in (W/'output/pdf').glob('*.pdf'):
 with pdfplumber.open(path) as pdf:
  pages=[]
  for i,p in enumerate(pdf.pages):
   assert abs(p.width-612)<.1 and abs(p.height-792)<.1
   body=[c for c in p.chars if c['top']<720] # Exclude bottom page-number footer.
   assert body
   bad=[c for c in body if c['x0']<71.5 or c['x1']>540.5 or c['top']<71.5 or c['bottom']>720.5]
   if bad:raise ValueError(f'Body exceeds one-inch margins: {path.name} page {i+1}, {bad[:2]}')
   minimum=min(c['size'] for c in body);assert minimum>=9.99
   pages.append({'page':i+1,'body_min_font_pt':minimum,'body_bounds':[min(c['x0'] for c in body),min(c['top'] for c in body),max(c['x1'] for c in body),max(c['bottom'] for c in body)]})
  checks.append({'file':path.name,'pages':len(pdf.pages),'sha256':sha(path),'page_checks':pages})
paper=PdfReader(W/'output/pdf/FVcell_Conference_Manuscript_Draft_20261027.pdf');refpage=next(i+1 for i,p in enumerate(paper.pages) if p.extract_text().splitlines()[1:2]==['References'])
assert refpage-1<=10
delivery=json.loads((W/'delivery_manifest.json').read_text(encoding='utf-8'))
assert refpage-1==delivery['main_pages']==10
blocks=json.loads((W/'manuscript_blocks.json').read_text(encoding='utf-8'))
assert len(paper.pages)==len(blocks) and refpage==11
source=json.loads((R/'available_study/completion.json').read_text());assert source['state']=='complete' and source['retrained_models']==0
rechecks=json.loads((R/'available_study/prediction_recheck.json').read_text());assert rechecks
for f in R.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
# Real matrix header, read only from a completely received leading range.
d=json.loads((R/'download_status.json').read_text());header=None
if d.get('state')=='complete':
 with gzip.open(R/'data/GSE278572_matrix.mtx.gz','rb') as f:
  header=f.readline().decode().strip();line=f.readline()
  while line.startswith(b'%'):line=f.readline()
  dims=list(map(int,line.split()))
 assert header=='%%MatrixMarket matrix coordinate integer general'
 assert dims[:2]==[36807,249799]
size=sum(f.stat().st_size for f in B.rglob('*') if f.is_file());assert size<100_000_000_000
primary=json.loads((R/'results/postrun_audit.json').read_text());assert primary['state']=='passed'
biology=B/'biological_strengthening_20261027'
source_files=json.loads((biology/'source_files_before.json').read_text(encoding='utf-8'))
for item in source_files:assert sha(item['path'])==item['sha256'],item['path']
case=json.loads((biology/'case_completion.json').read_text());assert case['state']=='complete' and case['no_model_refit']
assert sha(biology/'protocol_case_study.json')==case['source_protocol_sha256']
assert d['gzip_CRC_verified'] and d['verified_bytes']==3857480145
out={'state':'passed','original_protected_files_checked':len(manifest),'original_changed':changed,'body_page_limit':10,'actual_body_pages':refpage-1,'actual_pages_including_references_appendices':len(paper.pages),'PDF_format':'US Letter, 1-inch body margins, body/table/caption fonts >=10pt','pdf_checks':checks,'independent_study':'Replogle2022 K562/RPE1 complete; no new training','prediction_rechecks':len(rechecks),'primary_T_study':'GSE278572 complete: frozen-source transfer, all 4 conditions, 28 targets, 18103 genes; no donor/laboratory-disjoint claim','primary_T_postrun_audit':primary,'primary_T_actual_header':header,'primary_T_verified_download_bytes':d['verified_bytes'],'project_day_directory_logical_bytes':size,'total_cap_bytes':100_000_000_000,'latex_compile_status':'unavailable: Windows restricted-token sandbox; no terminal compiler found','PDF_authoring':'ReportLab; Poppler-rendered page QA','visual_review':'final rendered pages checked in contact sheet plus full-size critical pages','authors_pending':['English spelling confirmation','remaining authors','corresponding author']}
out.update(biological_source_files_unchanged=len(source_files),BCL9_case={'source_aggregates':case['observed_aggregates'],'held_donor_aggregates':case['held_donor_aggregates'],'no_model_refit':True,'post_hoc_not_independent_validation':True},references=delivery['references'],appendices=delivery['appendices'])
(W/'final_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='pdf_checks'},ensure_ascii=False,indent=2))
