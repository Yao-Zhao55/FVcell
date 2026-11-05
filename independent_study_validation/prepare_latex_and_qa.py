from pathlib import Path
import json,re,shutil,hashlib
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;B=R.parent;W=B/'conference_manuscript_20261027';L=W/'latex';(L/'images').mkdir(exist_ok=True)
blocks=json.loads((W/'manuscript_blocks.json').read_text(encoding='utf-8'))
def esc(s):
 mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','<':r'\ensuremath{<}','>':r'\ensuremath{>}'}
 return ''.join(mapping.get(c,c) for c in str(s)).replace('≤',r'\ensuremath{\leq}')
tex=[]
for j,pg in enumerate(blocks):
 if j:tex.append(r'\clearpage')
 for b in pg:
  k=b['kind'];s=b.get('text','')
  if k=='title':tex.append(r'\begin{center}{\Large\bfseries '+esc(s)+r'}\end{center}')
  elif k=='authors':tex.append(r'\begin{center}'+esc(s).replace('\n',r'\\')+r'\end{center}')
  elif k=='heading':tex.append(r'\subsection*{'+esc(s)+'}')
  elif k=='paragraph':tex.append(esc(s)+'\n')
  elif k=='equation':
   if s.startswith('delta_hat'):tex.append(r'\[\widehat{\delta}=\mu_{\mathrm{train}}+W_s h(u)+W_r\left[A(u)\odot\tanh\{S(z)+b_c+v_d\}\right].\]')
   else:tex.append(r'\[\mathrm{MAE}=\underset{d}{\mathrm{mean}}\ \underset{c}{\mathrm{mean}}\ \underset{a}{\mathrm{mean}}\ \underset{g}{\mathrm{mean}}\ |\delta_{dcag}-\widehat{\delta}_{dcag}|.\]')
  elif k=='figure':
   f=Path(b['path']);shutil.copy2(f,L/'images'/f.name);tex.append(r'\begin{center}\includegraphics[width=\linewidth]{images/'+f.name+r'}\end{center}'+ '\n'+esc(b['caption'])+'\n')
  elif k=='table':
   widths=b.get('widths') or [468/len(b['headers'])]*len(b['headers']);spec='@{}'+' '.join('>{\\raggedright\\arraybackslash}p{'+f'{v/468*.93:.3f}'+r'\linewidth}' for v in widths)+'@{}';tex.append(r'\begin{center}\begin{tabular}{'+spec+r'}\toprule');tex.append(' & '.join(esc(v) for v in b['headers'])+r'\\\midrule')
   for row in b['rows']:tex.append(' & '.join(esc(v) for v in row)+r'\\')
   tex.append(r'\bottomrule\end{tabular}\end{center}')
   if b.get('caption'):tex.append(esc(b['caption'])+'\n')
(L/'body.tex').write_text('\n'.join(tex),encoding='utf-8')
shutil.copy2(W/'references.bib',L/'references.bib')
# Inspect every page in contact sheets; critical pages are inspected separately.
imgs=sorted((W/'figures').glob('qa-manuscript-*.png'))+sorted((W/'figures').glob('qa-report-*.png'))
thumbw=280;thumbh=370;cols=4;rows=(len(imgs)+cols-1)//cols
sheet=Image.new('RGB',(cols*thumbw,rows*(thumbh+26)),'#ddd');d=ImageDraw.Draw(sheet)
for i,p in enumerate(imgs):
 im=Image.open(p).convert('RGB');im.thumbnail((thumbw-12,thumbh));x=(i%cols)*thumbw+(thumbw-im.width)//2;y=(i//cols)*(thumbh+26);sheet.paste(im,(x,y+22));d.text((i%cols*thumbw+8,y+4),p.stem,fill='black')
sheet.save(W/'figures/qa-contact-sheet.jpg',quality=90)
print('LaTeX source and QA sheet prepared:',len(imgs),'pages')
