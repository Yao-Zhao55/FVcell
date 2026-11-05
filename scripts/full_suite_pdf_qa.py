"""Render every report page and check text/geometry before delivery."""
from pathlib import Path
import subprocess,shutil,json
from PIL import Image,ImageDraw
import pdfplumber
BASE=Path(__file__).resolve().parents[1]/'full_experiments'
PDF=BASE/'output/pdf/FVcell_大纲v2模型实验报告_20261026.pdf'
QA=BASE/'output/pdf_previews';QA.mkdir(parents=True,exist_ok=True)
subprocess.run([shutil.which('pdftoppm'),'-r','100','-png',str(PDF),str(QA/'page')],check=True,capture_output=True)
pages=sorted(QA.glob('page-*.png'));checks=[]
with pdfplumber.open(PDF) as pdf:
    assert len(pages)==len(pdf.pages)
    for i,p in enumerate(pdf.pages):
        t=p.extract_text() or '';assert len(t)>100 and '\ufffd' not in t
        outside=[w['text'] for w in p.extract_words() if w['x0']<25 or w['x1']>p.width-25 or w['top']<20 or w['bottom']>p.height-12]
        assert not outside,(i+1,outside)
        checks.append({'page':i+1,'characters':len(t),'first_line':t.splitlines()[0],'text_geometry_within_page':True})
for offset in range(0,len(pages),6):
    sheet=Image.new('RGB',(760,1640),'#dce5e9');draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(pages[offset:offset+6]):
        im=Image.open(p).convert('RGB');im.thumbnail((360,510));x=(j%2)*380+10;y=(j//2)*545+25
        sheet.paste(im,(x,y));draw.text((x,y-18),'Page '+str(offset+j+1),fill='black')
    sheet.save(QA/f'overview-{offset//6+1}.jpg',quality=94)
(QA/'layout_checks.json').write_text(json.dumps({'pages':checks,'all_pages_rendered':True,'visual_review_still_required':True},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'pages':len(pages),'previews':str(QA)},ensure_ascii=False))
