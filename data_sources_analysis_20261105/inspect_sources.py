"""Read-only inspection of the newly supplied glioma experiment files."""

from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

from openpyxl import load_workbook
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data_sources"
PAPER = Path("D:/AI_virtual_cell_project/Digital_Organ_Twin/Di Zhu_Novel Small-MoleculeInhibitors.pdf")


def clean(value):
    if value is None:
        return None
    return str(value).replace("\n", " ").strip()[:180]


for path in sorted(SOURCES.iterdir()):
    if not path.is_file():
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"\nFILE {path.name} bytes={path.stat().st_size} sha256={digest}")
    if path.suffix.lower() == ".xlsx":
        wb = load_workbook(path, read_only=True, data_only=False)
        for ws in wb:
            print(f"SHEET {ws.title!r} rows={ws.max_row} cols={ws.max_column}")
            for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 32), values_only=True):
                print("ROW", json.dumps([clean(v) for v in row[:16]], ensure_ascii=False))
        wb.close()
    elif path.suffix.lower() == ".pzfx":
        tree = ET.parse(path)
        root = tree.getroot()
        for table in root.iter():
            if table.tag.rsplit("}", 1)[-1] != "Table":
                continue
            info = {k: v for k, v in table.attrib.items()}
            title = next((c.text for c in table if c.tag.rsplit("}", 1)[-1] == "Title"), None)
            print(f"TABLE {title!r} attributes={info}")
            for child in table:
                name = child.tag.rsplit("}", 1)[-1]
                if name in ("XColumn", "YColumn", "RowTitlesColumn"):
                    print("COLUMN", name, child.attrib, "TEXT", clean(" ".join((x.text or "") for x in child.iter() if x.text))[:240])
    elif path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        print("PAGES", len(reader.pages))
        for number, page in enumerate(reader.pages, 1):
            print("PAGE", number, clean(page.extract_text())[:2000])

print(f"\nPAPER {PAPER.name}")
reader = PdfReader(str(PAPER))
print("PAGES", len(reader.pages))
for number, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ""
    lines = [line.strip() for line in text.splitlines() if any(x in line.lower() for x in ("apopt", "ki-67", "ki67", "cell cycle", "prrx1", "pitx2", "icg-001", "1043", "c37-28r", "zd37-28r", "supplement"))]
    if lines:
        print("PAPER_PAGE", number, json.dumps(lines[:18], ensure_ascii=False))
