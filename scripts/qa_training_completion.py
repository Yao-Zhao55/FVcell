from pathlib import Path
import json,shutil,subprocess
import pdfplumber
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]/'training_completion'
PDF=ROOT/'output/pdf/FVcell_大纲训练补完核验_20261026.pdf';QA=ROOT/'output/pdf_previews';QA.mkdir(exist_ok=True)
for p in QA.glob('page-*.png'):
    assert p.resolve().is_relative_to(QA.resolve());p.unlink()
subprocess.run([shutil.which('pdftoppm'),'-r','100','-png',str(PDF),str(QA/'page')],capture_output=True,check=True)
pages=sorted(QA.glob('page-*.png'));checks=[]
with pdfplumber.open(PDF) as doc:
    assert len(doc.pages)==len(pages)
    for i,p in enumerate(doc.pages):
        text=p.extract_text() or '';assert len(text)>100 and '\ufffd' not in text
        outside=[w['text'] for w in p.extract_words() if w['x0']<24 or w['x1']>p.width-24 or w['top']<18 or w['bottom']>p.height-12];assert not outside,(i+1,outside)
        checks.append({'page':i+1,'text_characters':len(text),'first_line':text.splitlines()[0]})
for offset in range(0,len(pages),6):
    sheet=Image.new('RGB',(760,1640),'#dce5e9');draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(pages[offset:offset+6]):
        im=Image.open(p).convert('RGB');im.thumbnail((360,510));x=(j%2)*380+10;y=(j//2)*545+25;sheet.paste(im,(x,y));draw.text((x,y-18),'Page '+str(offset+j+1),fill='black')
    sheet.save(QA/f'overview-{offset//6+1}.jpg',quality=94)
(QA/'layout_checks.json').write_text(json.dumps({'all_pages_rendered':True,'pages':checks,'visual_review_required':True},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'pages':len(pages),'previews':str(QA)},ensure_ascii=False))
