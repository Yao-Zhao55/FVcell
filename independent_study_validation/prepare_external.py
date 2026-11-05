"""Stream complete public counts twice; never decompress the matrix to disk."""
from pathlib import Path
import sys,json,gzip,time,hashlib,csv,re,traceback,os
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'vendor_ml'))
import numpy as np,pandas as pd
S=R/'sources';D=R/'data';P=R/'prepared';P.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while a:=f.read(8*1024*1024):h.update(a)
 return h.hexdigest()
def status(stage,**kw):
 p=R/'preparation_status.tmp';p.write_text(json.dumps({'stage':stage,'pid':os.getpid(),'updated_epoch':time.time(),**kw},indent=2),encoding='utf-8');os.replace(p,R/'preparation_status.json')
def blocks(path):
 with gzip.open(path,'rb') as f:
  if f.readline().strip()!=b'%%MatrixMarket matrix coordinate integer general':raise ValueError('Not raw integer count matrix')
  while True:
   line=f.readline()
   if not line.startswith(b'%'):break
  dims=tuple(map(int,line.split()));yield dims
  tail=b'';entries=0
  while v:=f.read(16*1024*1024):
   v=tail+v;k=v.rfind(b'\n');tail=v[k+1:];a=np.fromstring(v[:k].decode('ascii'),sep=' ',dtype=np.int64)
   if len(a)%3:raise ValueError('Broken MatrixMarket record')
   a=a.reshape(-1,3);entries+=len(a);yield a
  if tail.strip():
   a=np.fromstring(tail.decode('ascii'),sep=' ',dtype=np.int64).reshape(-1,3);entries+=len(a);yield a
  if entries!=dims[2]:raise ValueError(f'NNZ mismatch {entries} != {dims[2]}')
  # Exhausting gzip verifies its stream CRC and uncompressed length.
def main():
 lock=json.loads((R/'FREEZE_LOCK.json').read_text());assert sha(R/'protocol_frozen.json')==lock['protocol_sha256']
 proto=json.loads((R/'protocol_frozen.json').read_text(encoding='utf-8'))
 state=json.loads((R/'download_status.json').read_text());assert state['state'] in ['download_complete_pending_gzip_crc','complete']
 matrix=D/'GSE278572_matrix.mtx.gz';assert matrix.stat().st_size==proto['download_bytes']
 features=pd.read_csv(S/'GSE278572_features.tsv.gz',sep='\t',header=None,names=['id','name','type'])
 barcodes=pd.read_csv(S/'GSE278572_barcodes.tsv.gz',header=None)[0].astype(str).to_numpy();n=len(barcodes)
 calls=pd.read_csv(S/'GSE278572_protospacer_calls_per_cell.csv.gz');calls=calls[calls.num_features.eq(1)].set_index('cell_barcode')
 if not calls.index.is_unique:raise ValueError('Duplicate guide assignments')
 guide=calls.feature_call.reindex(barcodes).fillna('UNASSIGNED').to_numpy();guides=sorted(set(guide)-{'UNASSIGNED'});gmap={g:i for i,g in enumerate(guides)}
 gi=np.array([gmap.get(g,-1) for g in guide],np.int32)
 isgex=features.type.eq('Gene Expression').to_numpy();ismito=features.name.str.startswith('MT-').to_numpy()&isgex
 htoidx=[int(features.index[features.id.eq('HTO_'+str(k))][0]) for k in range(1,5)]
 lib=np.zeros(n,np.int64);ng=np.zeros(n,np.int32);mito=np.zeros(n,np.int64);hto=np.zeros((4,n),np.int64)
 if (P/'cell_qc.npz').exists():
  q=np.load(P/'cell_qc.npz');lib=q['library'];ng=q['n_genes'];mito=q['mito'];hto=q['hto']
 else:
  stream=blocks(matrix);dims=next(stream);assert dims[:2]==(len(features),n);done=0;status('QC_stream',matrix_dimensions=dims)
  for a in stream:
   row=a[:,0]-1;col=a[:,1]-1;v=a[:,2]
   if (v<0).any() or (row<0).any() or (row>=len(features)).any() or (col<0).any() or (col>=n).any():raise ValueError('Invalid count/index')
   m=isgex[row];np.add.at(lib,col[m],v[m]);np.add.at(ng,col[m],v[m]>0)
   m=ismito[row];np.add.at(mito,col[m],v[m])
   for k,j in enumerate(htoidx):
    m=row==j;np.add.at(hto[k],col[m],v[m])
   done+=len(a);status('QC_stream',entries_done=done,entries_total=dims[2])
  np.savez_compressed(P/'cell_qc.npz',library=lib,n_genes=ng,mito=mito,hto=hto)
 order=np.sort(hto,axis=0);tag=hto.argmax(0);ratio=order[-1]/np.maximum(order[-2],1)
 ok=(gi>=0)&(lib>=1000)&(ng>=500)&(mito/np.maximum(lib,1)<=.2)&(order[-1]>=50)&(ratio>=3)
 groups=np.where(ok,gi*4+tag,-1);ncell=np.bincount(groups[ok],minlength=len(guides)*4).reshape(-1,4)
 feats=features[isgex].copy();gx=np.full(len(features),-1,np.int32);gx[np.flatnonzero(isgex)]=np.arange(isgex.sum())
 counts=np.zeros((len(guides)*4,isgex.sum()),np.int64)
 if (P/'guide_pseudobulk.npz').exists():counts=np.load(P/'guide_pseudobulk.npz')['counts'].reshape(counts.shape)
 else:
  stream=blocks(matrix);dims=next(stream);done=0
  for a in stream:
   row=a[:,0]-1;col=a[:,1]-1;v=a[:,2];gg=groups[col];gene=gx[row];m=(gg>=0)&(gene>=0)
   np.add.at(counts.ravel(),gg[m]*counts.shape[1]+gene[m],v[m]);done+=len(a);status('aggregate_stream',entries_done=done,entries_total=dims[2])
  np.savez_compressed(P/'guide_pseudobulk.npz',counts=counts.reshape(len(guides),4,-1),n_cells=ncell,guides=np.array(guides),gene_ids=np.asarray(feats.id,dtype=str),gene_names=np.asarray(feats.name,dtype=str))
 manifest={'source_sha256':state['sha256'],'gzip_CRC_verified':True,'matrix_features':len(features),'matrix_cells':n,'single_guide_cells':int((gi>=0).sum()),'QC_cells':int(ok.sum()),'hashtag_features':[features.loc[j,['id','name']].to_dict() for j in htoidx],'QC_cells_by_hashtag':{features.loc[htoidx[k],'name']:int((ok&(tag==k)).sum()) for k in range(4)},'guides':guides,'n_GEX_features':int(isgex.sum()),'all_GEX_in_library_denominator':True,'conditions_mixed_donors':True,'protocol_sha256':lock['protocol_sha256']}
 (P/'count_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 state['state']='complete';state['gzip_CRC_verified']=True;state['updated_epoch']=time.time();(R/'download_status.json').write_text(json.dumps(state,indent=2))
 status('complete',QC_cells=int(ok.sum()),guides=len(guides));print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=str(e),traceback=traceback.format_exc());raise
