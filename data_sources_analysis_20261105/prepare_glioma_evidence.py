"""Extract supplied Prism/plate data without altering the source files.

The source-level C37-28R -> publication-level ZD37-28R mapping was confirmed
by the data provider in the conversation on 2026-11-05. Replicate entries are
not assumed to be independent biological experiments without a sample map.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from openpyxl import load_workbook
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "data_sources"
PAPER = Path("D:/AI_virtual_cell_project/Digital_Organ_Twin/Di Zhu_Novel Small-MoleculeInhibitors.pdf")


def local_name(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def children(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in node if local_name(child) == name]


def label(node: ET.Element) -> str:
    titles = children(node, "Title")
    return "".join(titles[0].itertext()).strip() if titles else ""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


prism_rows: list[dict] = []
summary_rows: list[dict] = []
counts: dict[str, int] = {}
for filename, assay in [
    ("Apoptosis.pzfx", "Annexin V/PI apoptotic-cell percentage"),
    ("cellcycle.pzfx", "cell-cycle percentage; phase unspecified in file"),
    ("EMT-QPCR.pzfx", "relative qPCR value as stored"),
]:
    root = ET.parse(SRC / filename).getroot()
    for table in root.iter():
        if local_name(table) != "Table":
            continue
        table_title = label(table)
        if table_title == "Data 3":
            # A duplicated/partial working table, not a third biological context.
            continue
        for group in children(table, "YColumn"):
            group_label = label(group)
            values: list[float] = []
            for index, subcol in enumerate(children(group, "Subcolumn"), 1):
                entries = [float(x.text) for x in subcol.iter() if local_name(x) == "d" and x.text and x.text.strip()]
                for value in entries:
                    prism_rows.append({
                        "source_file": filename,
                        "assay_from_filename": assay,
                        "table": table_title,
                        "group_as_stored": group_label,
                        "replicate_position": index,
                        "value_as_stored": value,
                    })
                    values.append(value)
            if values:
                summary_rows.append({
                    "source_file": filename,
                    "assay_from_filename": assay,
                    "table": table_title,
                    "group_as_stored": group_label,
                    "n_nonmissing_entries": len(values),
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else "",
                    "min": min(values),
                    "max": max(values),
                })
    counts[filename] = sum(1 for row in prism_rows if row["source_file"] == filename)

write_csv(HERE / "prism_observations.csv", prism_rows,
          ["source_file", "assay_from_filename", "table", "group_as_stored", "replicate_position", "value_as_stored"])
write_csv(HERE / "prism_descriptive_summary.csv", summary_rows,
          ["source_file", "assay_from_filename", "table", "group_as_stored", "n_nonmissing_entries", "mean", "sample_sd", "min", "max"])

plate_rows: list[dict] = []
for path in sorted(SRC.glob("*.xlsx")):
    workbook = load_workbook(path, read_only=True, data_only=True)
    for sheet in workbook:
        for row in sheet.iter_rows(min_row=28):
            row_label = row[1].value if len(row) > 1 else None
            if not isinstance(row_label, str) or row_label not in "ABCDEFGH":
                continue
            for cell in row[2:14]:
                if isinstance(cell.value, (int, float)):
                    plate_rows.append({
                        "source_file": path.name,
                        "sheet": sheet.title,
                        "well": cell.coordinate,
                        "row_label": row_label,
                        "column_number": cell.column - 2,
                        "absorbance_450_nm": cell.value,
                        "treatment": "unmapped",
                        "dose": "unmapped",
                        "sample_or_experiment_id": "unmapped",
                    })
    workbook.close()
write_csv(HERE / "plate_wells_unmapped.csv", plate_rows,
          ["source_file", "sheet", "well", "row_label", "column_number", "absorbance_450_nm", "treatment", "dose", "sample_or_experiment_id"])

source_files = sorted(path for path in SRC.iterdir() if path.is_file())
manifest = {
    "mapping_attestation": "User confirmed C37-28R in the Prism files is ZD37-28R in the published glioma article on 2026-11-05.",
    "article": {"title": "Pharmacological Study of Novel Small-Molecule Inhibitors Targeting B-Cell Lymphoma 9/beta-Catenin in the Treatment of Glioma", "doi": "10.1021/acsptsci.6c00284", "local_pdf_sha256": digest(PAPER)},
    "sources": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in source_files},
    "prism_entry_counts": counts,
    "plate_numeric_wells": {p.name: sum(row["source_file"] == p.name for row in plate_rows) for p in SRC.glob("*.xlsx")},
    "Ki67_pdf_pages": {p.name: len(PdfReader(str(p)).pages) for p in SRC.glob("*.pdf")},
    "key_limitations": [
        "Prism entries lack biological replicate and batch identifiers; no inferential P values are recalculated.",
        "Cell-cycle file lacks a phase or endpoint label; its percentages are not assigned to G1/S/G2-M.",
        "Plate workbooks lack a well-to-treatment/dose map, a blank-control map and experiment identifiers; IC50 is not refitted.",
        "Ki67 PDFs contain selected gate plots, not FCS events or complete replicate tables.",
        "These glioma drug assays are not held-out predictions by the primary-T CRISPRi FVcell model.",
    ],
}
(HERE / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"prism_entry_counts": counts, "prism_groups": len(summary_rows), "plate_numeric_wells": manifest["plate_numeric_wells"], "source_sha256": {name: rec["sha256"] for name, rec in manifest["sources"].items()}}, ensure_ascii=False, indent=2))
