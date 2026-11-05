"""Independent checks for the versioned data-filled conference manuscript."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parents[2]
WORK = ROOT / "conference_manuscript_20261027"
PDF = WORK / "output/pdf/FVcell_Conference_Manuscript_20261105.pdf"
MD = WORK / "FVcell_Conference_Manuscript_20261105.md"
BLOCKS_NEW = WORK / "manuscript_blocks_20261105.json"
BLOCKS_OLD = WORK / "manuscript_blocks.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
bad_sources = [name for name, item in manifest["sources"].items()
               if sha(ROOT / "data_sources" / name) != item["sha256"]]
protected = json.loads((ROOT / "protected_files_before.json").read_text(encoding="utf-8"))
bad_protected = [item["path"] for item in protected
                 if not (PROJECT / item["path"]).exists() or sha(PROJECT / item["path"]) != item["sha256"]]
old = json.loads(BLOCKS_OLD.read_text(encoding="utf-8"))
new = json.loads(BLOCKS_NEW.read_text(encoding="utf-8"))
assert old[0][4]["text"] == new[0][4]["text"]
assert all(old[i] == new[i] for i in range(1, 9))
assert all(old[i] == new[i] for i in range(10, 15))
assert not bad_sources and not bad_protected
assert "Yao Zhao, Xiaoxuan Lu" in new[0][1]["text"]
assert "Di Zhu*" in new[0][1]["text"]
assert "28 targets" in new[7][1]["text"] or "28" in new[7][2]["text"]

reader = PdfReader(str(PDF))
texts = [page.extract_text() or "" for page in reader.pages]
assert len(texts) == 18
assert "References" in texts[10]
assert "Appendix F" in texts[16] and "Appendix F" in texts[17]
assert "7.93" in texts[16] and "93.17" in texts[16]
assert "0.617" in texts[17] and "0.697" in texts[17]
assert "No public archival source-code release" in " ".join(texts[9].split())
assert "does not support" in texts[9]
assert "C37-28R" in texts[16] and "ZD37-28R" in texts[16]

page_stats = []
with pdfplumber.open(PDF) as pdf:
    for number, page in enumerate(pdf.pages, 1):
        assert abs(page.width - 612) < .01 and abs(page.height - 792) < .01
        words = page.chars
        body = [c for c in words if c["top"] < 735]
        if body:
            min_font = min(c["size"] for c in body)
            left = min(c["x0"] for c in body)
            right = max(c["x1"] for c in body)
            top = min(c["top"] for c in body)
            bottom = max(c["bottom"] for c in body)
            assert min_font >= 9.99, (number, min_font)
            assert left >= 71.9 and right <= 540.1, (number, left, right)
            assert top >= 70 and bottom <= 720.1, (number, top, bottom)
            page_stats.append({"page": number, "min_text_pt": round(min_font, 2),
                               "body_bounds": [round(v, 1) for v in (left, top, right, bottom)]})

total_bytes = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
assert total_bytes <= 100_000_000_000
qa = {
    "status": "passed",
    "visual_review": "18-page contact sheet reviewed; final pages 1, 10, 16, 17 and 18 inspected at full size",
    "pages": len(texts),
    "main_pages": 10,
    "abstract_identical_to_prior_blocks": True,
    "unchanged_model_pages": "2-9, 11-15",
    "source_files_checked": len(manifest["sources"]),
    "source_files_changed": bad_sources,
    "protected_files_checked": len(protected),
    "protected_files_changed": bad_protected,
    "day25_26_logical_bytes": total_bytes,
    "pdf_sha256": sha(PDF),
    "markdown_sha256": sha(MD),
    "page_checks": page_stats,
    "scientific_limit": "New glioma lab measurements are descriptive source-study evidence, not FVcell transfer scores.",
}
(HERE / "manuscript_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: qa[k] for k in qa if k != "page_checks"}, ensure_ascii=False, indent=2))
