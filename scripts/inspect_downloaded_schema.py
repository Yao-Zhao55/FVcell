"""Read HDF5 metadata only from completed, receipted public data files."""
from pathlib import Path
import json
from run_first_training import h5py,column,index,matrix_shape
BASE=Path(__file__).resolve().parents[1]
manifest=json.loads((BASE/'large_asset_manifest.json').read_text(encoding='utf-8'))
result={}
for key in ('de_stats','pseudobulk'):
    p=BASE/'large_data'/Path(manifest[key]['key']).name
    receipt=p.with_name(p.name+'.verified.json')
    if not (p.exists() and receipt.exists()):continue
    r=json.loads(receipt.read_text(encoding='utf-8'))
    if p.stat().st_size!=manifest[key]['bytes'] or r['etag']!=manifest[key]['etag']:raise ValueError('Invalid completed source')
    with h5py.File(p,'r') as f:
        record={'root_keys':list(f.keys()),'obs_columns':list(f['obs'].keys()),'obs_count':len(index(f['obs'])),'gene_count':len(index(f['var'])),'gene_examples':index(f['var'])[:5].tolist(),'layers':{name:{'shape':matrix_shape(node),'encoding':str(node.attrs.get('encoding-type','dense'))} for name,node in f.get('layers',{}).items()},'metadata_examples':{}}
        for name in ('culture_condition','target_contrast','target','donor_id','perturbed_gene_id','perturbed_gene_name','guide_type'):
            if name in f['obs']:
                values=column(f['obs'][name]);unique=list(dict.fromkeys(str(v) for v in values));record['metadata_examples'][name]={'unique_count':len(unique),'examples':unique[:8],'control_like_values':[v for v in unique if any(token in v.upper() for token in ('TARGET','CONTROL','NTC'))][:20]}
        if 'X' in f:record['X']={'shape':None if isinstance(f['X'],h5py.Dataset) and f['X'].shape is None else matrix_shape(f['X']),'encoding':str(f['X'].attrs.get('encoding-type','dense'))}
        result[key]=record
(BASE/'prepared/downloaded_hdf5_schema.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
