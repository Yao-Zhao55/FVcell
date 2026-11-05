# Data availability and attribution

| Role | Original work / provider | Public source | Included here |
| --- | --- | --- | --- |
| Source training and within-study holdouts | Ronghui Zhu et al., *Cell* 2026, DOI 10.1016/j.cell.2026.08.002 | [GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342); [author data-sharing repository](https://github.com/emdann/GWT_perturbseq_analysis_2025) | Derived row/feature manifests, frozen splits, checksums and scalar scores; no original HDF5 |
| External cell-line audit | Joseph M. Replogle et al., *Cell* 2022, DOI 10.1016/j.cell.2022.05.013 | [scPerturb Zenodo record 13350497](https://zenodo.org/records/13350497), K562 essential and RPE1 matrices | Scalar evaluation and receipt files; no original H5AD |
| Primary-T external audit | Maya M. Arce et al., *Nature* 2025, DOI 10.1038/s41586-024-08314-y | [GSE278572](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278572) | Frozen protocol, count provenance, aggregate evaluation and integrity receipt; no matrix.mtx.gz |
| BCL9/BCL9L mechanism illustration | Yuan-yuan Zhu et al., DOI 10.1038/s41401-026-01845-2 | Published paper | Source-study RNA case summaries only |
| Glioma drug-assay context | Huiyu Li et al., DOI 10.1021/acsptsci.6c00284 | Published paper; separate group-supplied underlying assays | Citation and parsing code only; supplied Prism, plate-well and PDF source files are withheld pending rights/metadata confirmation |

The supplied group assays are not FVcell predictions. Their raw files and
individual well/Prism values are not in this public package. Missing plate maps,
replicate IDs and paired single-cell data preclude a drug-response validation.
No individual patient record is knowingly included. The source studies retain
their own licenses and attribution; this repository does not relicense them.

The two GWT HDF5 inputs total 61,352,897,247 bytes. The primary-T count archive
is 3,857,480,145 bytes. GitHub is used for code, manifests and results, not
for these source matrices. The GWT SHA256 receipts are in `source_receipts/`.
The Arce archive has a locally checked SHA256 and gzip CRC, but no publisher
checksum is claimed. Replogle files are tied to the Zenodo MD5 receipts.
