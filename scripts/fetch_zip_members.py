"""Read selected tiny members using HTTP byte ranges; validate ZIP CRC32."""
from pathlib import Path
import json,subprocess,shutil,struct,zlib,hashlib
BASE=Path(__file__).resolve().parents[1]
URL='https://zenodo.org/api/records/13924126/files/data_tables.zip/content'
entries=json.loads((BASE/'zip_entry_index.json').read_text(encoding='utf-8'))
out=BASE/'sources'/'selected_tables';out.mkdir(exist_ok=True)
reports=[]
for item in entries:
    if not item['name'].startswith('data_tables/') or not any(t in item['name'] for t in ['S8_','S9_']):continue
    if item['compressed']>131072:raise RuntimeError('Unexpected large member')
    fragment=out/(Path(item['name']).name+'.fragment')
    headers=out/(Path(item['name']).name+'.headers')
    start=item['offset'];end=start+item['compressed']+1023
    result=subprocess.run([shutil.which('curl') or 'curl','-L','--fail','--silent','--show-error','--connect-timeout','15','--max-time','45','--max-filesize','132096','--range',f'{start}-{end}',URL,'-D',str(headers),'-o',str(fragment)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    if '206' not in headers.read_text(errors='replace'):raise RuntimeError('Server did not confirm byte range')
    data=fragment.read_bytes()
    h=struct.unpack_from('<4s5H3L2H',data,0)
    if h[0]!=b'PK\x03\x04':raise RuntimeError('Missing local ZIP header')
    offset=30+h[9]+h[10]
    compressed=data[offset:offset+item['compressed']]
    raw=zlib.decompress(compressed,-15) if item['method']==8 else compressed
    if len(raw)!=item['bytes'] or zlib.crc32(raw)!=item['crc32']:raise RuntimeError('ZIP member checksum failure')
    p=out/Path(item['name']).name;p.write_bytes(raw)
    reports.append({**item,'file':str(p.relative_to(BASE)),'sha256':hashlib.sha256(raw).hexdigest(),'validation':'published-version HTTP range + central/local ZIP entry + member CRC32; full archive MD5 not checked','downloaded_fragment_bytes':len(data)})
    print(json.dumps(reports[-1],ensure_ascii=False),flush=True)
(BASE/'selected_table_checksums.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
