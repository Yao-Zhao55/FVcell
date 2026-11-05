"""Check new inference against saved old biological test-row scores."""
from evaluate_frozen import *
def main():
 genes=pd.read_csv(B/'full_experiments/prepared/donor_genes.csv').gene_id.astype(str).to_numpy()
 ctrl=np.load(B/'full_experiments/prepared/controls.npz');lookup={(str(d),str(c)):i*3+j for i,d in enumerate(ctrl['donors']) for j,c in enumerate(ctrl['contexts'])}
 y=np.load(B/'full_experiments/prepared/donor_effects.npy',mmap_mode='r');checks=[]
 for fam in ['hierarchical','fixed_gaussian','fv_continued','concat_original_P1']:
  task=B/'training_completion/runs/hierarchical_donor_fold0' if fam!='concat_original_P1' else B/'full_experiments/runs/E1_donor_P1'
  folder=task/(fam if fam!='concat_original_P1' else 'concat')/'11';m=pd.read_csv(task/'rows.csv');m=m[m.partition.eq('test')].head(12).copy().reset_index(drop=True)
  control=np.stack([ctrl['control_log2cpm'][lookup[r.donor_id,r.culture_condition]] for r in m.itertuples()]);pred,ck=checkpoint_predictions(folder/'model.pt',m.target_contrast.to_numpy(),m.culture_condition.to_numpy(),control,genes)
  measured=np.asarray(y[m.storage_row.to_numpy()]);metric=np.abs(measured.astype(float)-pred.astype(float)).mean(1);saved=pd.read_csv(folder/'test_rows.csv').set_index('index');expected=saved.loc[m['index'],'mae'].to_numpy();error=float(np.max(np.abs(metric-expected)))
  if error>2e-6:raise ValueError(f'Frozen inference mismatch {fam}: {error}')
  checks.append({'model':fam,'checked_real_rows':len(m),'max_MAE_difference_from_existing_score':error,'pass':True})
 save('inference_verification.json',checks);print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
