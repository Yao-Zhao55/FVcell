# FVcell: evidence access and transfer limits

This repository accompanies *FVcell: Evidence Access and Transfer Limits in
T-Cell Perturbation Prediction* by Yao Zhao, Xiaoxuan Lu, Yang Yang and Di Zhu.
It is a research evaluation of a shared-response/context-residual model.
The results include negative findings and do not support clinical or organ-level
prediction claims.

## Navigate the release

- `scripts/`: training, baselines, splits, tests and analysis code from the
  companion code package.
- `independent_study_validation/`: frozen source-model evaluation on GSE278572
  and the Replogle K562/RPE1 study, plus all scalar outputs.
- `prepared/`, `full_experiments/`, `training_completion/analysis/`:
  frozen split rules, row/feature manifests, experiment protocols and results.
- `paper_results/`: complete paper-referenced scalar tables and bibliography.
- `figures/`: plots derived from public data and computational experiments.
- `source_receipts/`: download sizes, hashes and integrity checks.
- `DATA_AVAILABILITY.md`: upstream sources and data boundaries.
- `REPRODUCIBILITY.md`: runnable order and limitations.

## Quick software check

Install the dependencies listed in `requirements.txt` for your operating
system and PyTorch/CUDA setup. From the repository root, run:

```sh
cd scripts
python -m unittest test_split_rules test_first_training test_atomic_json test_full_suite
```

These checks exercise code paths with small fixtures; they are not the
biological experiments. Training requires the separately downloaded public
matrices and the frozen protocol described in `REPRODUCIBILITY.md`.

The source code, frozen protocols and scalar results are provided as
separate browseable files in this repository. Public source matrices
remain at their original repositories; see `DATA_AVAILABILITY.md`.
