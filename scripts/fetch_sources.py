"""Fetch public source metadata/small tables into this isolated work package."""
from pathlib import Path
import subprocess,json,hashlib,sys,shutil,xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
BASE=Path(__file__).resolve().parents[1]
RAW=BASE/'sources';RAW.mkdir(parents=True,exist_ok=True)
LOG=BASE/'fetch_manifest.json'
records=json.loads(LOG.read_text(encoding='utf-8')) if LOG.exists() else []
def fetch(item):
    name,url=item
    path=RAW/name
    if path.exists() and path.stat().st_size:
        return {'name':name,'url':url,'status':'cached','bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    result=subprocess.run([shutil.which('curl') or 'curl','-L','--fail','--retry','1','--connect-timeout','15','--max-time','150','--max-filesize','134217728','--silent','--show-error',url,'-o',str(path)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    entry={'name':name,'url':url,'status':'ok' if result.returncode==0 else 'failed','error':result.stderr[-1000:]}
    if result.returncode==0:
        data=path.read_bytes();entry.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    elif path.exists():path.unlink() # Only this run's incomplete download.
    return entry
if len(sys.argv)>1 and sys.argv[1]=='fallback':
    items=[
      ('GWT_DE_stats_mirror.csv','https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/DE_stats.suppl_table.csv'),
      ('data_tables_direct.zip','https://zenodo.org/records/13924126/files/data_tables.zip?download=1'),
      ('gwt_experiments_config_mirror.yaml','https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/experiments_config.yaml')]
elif len(sys.argv)>1 and sys.argv[1]=='gwt':
    xml=ET.fromstring((RAW/'gwt_s3_listing.xml').read_bytes())
    ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
    keys=[c.find('s:Key',ns).text for c in xml.findall('s:Contents',ns)]
    wanted={'DE_stats.suppl_table.csv','sample_metadata.suppl_table.csv','validation_donor_metadata.csv','IL10_IL21_arrayed_validation_combined.csv','proliferation_stats_IL10IL21.csv','Th1Th2_validation_summary.suppl_table.csv'}
    selected_keys=[k for k in keys if k.split('/')[-1] in wanted]
    items=[('GWT_'+k.split('/')[-1],'https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/'+k) for k in selected_keys]
    items.append(('gwt_experiments_config.yaml','https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/experiments_config.yaml'))
elif len(sys.argv)>1 and sys.argv[1]=='tables':
    meta=json.loads((RAW/'13924126_record.json').read_text(encoding='utf-8'))
    selected=[x for x in meta['files'] if x['key'] in ['data_tables.zip','Main_figure_scripts.zip']]
    items=[(x['key'],x['links']['self']) for x in selected]
else:
    items=[
      ('13924126_record.json','https://zenodo.org/api/records/13924126'),
      ('gwt_data_sharing_readme.md','https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/data_sharing_readme.md'),
      ('gwt_repository_tree.json','https://api.github.com/repos/emdann/GWT_perturbseq_analysis_2025/git/trees/master?recursive=1'),
      ('gwt_s3_listing.xml','https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/?list-type=2&prefix=marson2025_data/'),
      ('GSE278572_series.txt','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278572&targ=self&form=text&view=full'),
      ('GSE314342_series.txt','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342&targ=self&form=text&view=full'),
      ('GSE299043_family.soft.gz','https://ftp.ncbi.nlm.nih.gov/geo/series/GSE299nnn/GSE299043/soft/GSE299043_family.soft.gz')]
with ThreadPoolExecutor(max_workers=3) as pool:
    for entry in pool.map(fetch,items):
        records=[r for r in records if r['name']!=entry['name']]+[entry]
        LOG.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(entry,ensure_ascii=False),flush=True)
if len(sys.argv)>1 and sys.argv[1]=='tables':
    checks=[]
    for item in selected:
        p=RAW/item['key']
        checks.append({'name':item['key'],'expected':item['checksum'],'actual':'md5:'+hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None})
    (BASE/'download_checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    if any(c['expected']!=c['actual'] for c in checks):raise RuntimeError('Published checksum mismatch')
