# Reproducibility and evaluation scope

Use Python 3.12; package versions are in `requirements.txt` from the companion
code package and `full_experiments/software_and_code_manifest.json`. CUDA 12.8
and an RTX 5060 Laptop GPU were recorded locally. Equivalent hardware is not
guaranteed to reproduce floating-point bits or run time.

1. Obtain the public source data from `DATA_AVAILABILITY.md` and verify the
   named receipts. Keep raw matrices out of Git.
2. Read `prepared/FREEZE_LOCK.json`, the split CSVs and the task-specific
   protocol JSONs before fitting. Do not tune on external test outcomes.
3. Use the matching `scripts/` entries described in
   `full_experiments/EXPERIMENTS_REPRODUCE.md` for source training, strong
   baselines, four donor holdouts, analysis and audit.
4. Run `independent_study_validation/prepare_external.py` and then
   `evaluate_frozen.py` only after obtaining the public count archive and
   reproducing the source checkpoints. The external donors were pooled; this
   is not donor-disjoint confirmation.
5. Inspect `paper_results/` and the full `independent_study_validation/results/`
   tables. Report all predefined backgrounds and three model seeds, including
   unfavorable results. The main pooled task uses a different label unit from
   count-derived external contrasts; scores must not be placed in one ranking.

The small `mechanism.npz`, `mechanism_randomized.npz` and `controls.npz` files
are derived feature/control fixtures used by the code and tests. They do not
replace the full response matrices. Frozen model weights are **not** included.
The public source and split rules permit retraining, but exact frozen-weight
inference requires the original checkpoint hashes listed in the manifests or
a later separately reviewed checkpoint release. Official STATE code must be
obtained separately from https://github.com/ArcInstitute/state under its own
license; the local endpoint adapter is not the published STATE training method.

Software-check fixtures are synthetic and are not biological findings. The
reported external comparisons are exploratory where original outcomes were
previously inspected. The manuscript does not claim algorithmic superiority.
