# What the public tables support

- On pooled held-out combinations, FVcell MAE is 0.173411; concatenation is
  0.173536; a shrunk additive baseline is 0.172030. There is no pooled gain
  over that baseline.
- In external Replogle K562/RPE1, FVcell MAE is 0.232618 versus 0.227809 for
  zero response. This is a study and cell-background shift.
- In GSE278572, the prespecified Teff backgrounds give hierarchical MAE
  0.130631 versus 0.113051 for zero response and 0.126337 for the source-only
  additive baseline. The two donors were pooled before sequencing.
- BCL9/BCL9L RNA modules are an exploratory public-study illustration.
  Supplied glioma drug assays are separately attributed and do not validate
  FVcell drug or organ prediction.

The repository supports inspection of the reported analysis. It must not be
presented as a validated patient-level digital twin, a confirmed new method
advantage, or a reproduced drug-response model.
