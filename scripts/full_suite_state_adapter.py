"""Official STATE pseudobulk backbone, explicitly adapted to endpoint response.

Source: ArcInstitute/state, CC-BY-NC-SA-4.0, pinned blobs in upstream_state.
No pretrained parameters, artificial cells, OT/distributional accuracy claim.
Only optional unused VCI decoder import is made lazy in a separate runtime copy.
"""
from pathlib import Path
import sys,types,importlib,hashlib,json,os
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'full_experiments'
sys.path.insert(0,str(BASE/'vendor_state'));sys.path.insert(0,str(BASE/'vendor_ml'));sys.path.insert(0,str(BASE/'vendor'))
import torch
from torch import nn
def load_official():
    original=ROOT/'upstream_state/src/state/tx/models';runtime=ROOT/'state_runtime'/str(os.getpid())/'models';runtime.mkdir(parents=True,exist_ok=True)
    changes=[]
    for name in ('base.py','utils.py','decoders.py','pseudobulk.py'):
        data=(original/name).read_text(encoding='utf-8');modified=data
        if name=='decoders.py':
            modified=modified.replace('from ...emb.finetune_decoder import Finetune','')
            marker='        self.finetune = Finetune('
            modified=modified.replace(marker,'        from ...emb.finetune_decoder import Finetune\n'+marker)
        (runtime/name).write_text(modified,encoding='utf-8')
        changes.append({'file':name,'original_sha256':hashlib.sha256(data.encode()).hexdigest(),'runtime_sha256':hashlib.sha256(modified.encode()).hexdigest(),'change':'lazy optional Finetune import only' if modified!=data else 'none'})
    for name,path in [('_fvstate',ROOT/'state_runtime'),('_fvstate.tx',ROOT/'state_runtime'),('_fvstate.tx.models',runtime)]:
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    klass=importlib.import_module('_fvstate.tx.models.pseudobulk').PseudobulkPerturbationModel
    (ROOT/'state_adapter_provenance.json').write_text(json.dumps({'name':'STATE official pseudobulk backbone, endpoint adapter','source':'https://github.com/ArcInstitute/state','license':'CC-BY-NC-SA-4.0','modifications':changes,'adaptations':['sequence length 1 = one real aggregated endpoint, not a fake single cell','input control embedding 35D, perturbation features 65D, identical permissions to other suite models','signed logFC response output; no pretrained decoder or weights','same supervised robust endpoint loss and budget as matched baselines, not original OT set-distribution training'],'not_a_replication_of_published_STATE_single_cell_metrics':True},indent=2),encoding='utf-8')
    return klass
class StateAdapter(nn.Module):
    def __init__(self,g):
        super().__init__();klass=load_official()
        self.network=klass(input_dim=35,hidden_dim=64,output_dim=g,pert_dim=65,gene_dim=g,predict_residual=False,output_space='embedding',embed_key='endpoint_response',gene_decoder_bool=False,dropout=.1,loss='mse',n_encoder_layers=2,n_decoder_layers=2,transformer_backbone_kwargs={'n_embd':64,'n_layer':2,'n_head':4,'n_positions':1,'n_ctx':1,'vocab_size':1,'use_cache':False})
    def forward(self,u,z,c=None):
        value=self.network({'pert_emb':u,'ctrl_cell_emb':z},padded=True)
        return value,u,u
if __name__=='__main__':
    m=StateAdapter(100).cuda();u=torch.randn(8,65,device='cuda');z=torch.randn(8,35,device='cuda');y=m(u,z)[0];y.square().mean().backward();print(json.dumps({'forward_backward':'passed','shape':list(y.shape),'parameters':sum(p.numel() for p in m.parameters())}))
