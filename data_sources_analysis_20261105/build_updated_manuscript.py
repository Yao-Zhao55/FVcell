"""Build a versioned FVcell draft with audited glioma assay evidence.

The registered abstract and all frozen model results are copied verbatim.
This script never rewrites the prior working manuscript or source data.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor_plot"))

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "conference_manuscript_20261027"
OUT = WORK / "output" / "pdf"
LATEX = WORK / "latex_20261105"
FIG = WORK / "figures"
BLOCKS = WORK / "manuscript_blocks.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_summary() -> dict[tuple[str, str, str], dict]:
    with (HERE / "prism_descriptive_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(r["source_file"], r["table"], r["group_as_stored"]): r for r in rows}


def block(kind: str, text: str) -> dict:
    return {"kind": kind, "text": text}


def table(headers: list[str], rows: list[list[str]], widths: list[int], caption: str) -> dict:
    return {"kind": "table", "headers": headers, "rows": rows, "widths": widths, "caption": caption}


def make_plot(summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 2.35))
    doses = [0, 2.5, 5, 10]
    for cell, color in [("U87-MG", "#24638b"), ("U251", "#a55a31")]:
        means, sds = [], []
        for dose in doses:
            label = "Control" if dose == 0 else f"{dose:g} μM C37-28R"
            row = summary[("Apoptosis.pzfx", cell, label)]
            means.append(float(row["mean"]))
            sds.append(float(row["sample_sd"]))
        ax.errorbar(doses, means, yerr=sds, marker="o", lw=1.5, capsize=3, color=color, label=cell)
    ax.set_xlabel("Concentration (μM; 0 = control)")
    ax.set_ylabel("Apoptotic cells (%)")
    ax.set_xticks(doses)
    ax.set_ylim(0, 103)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e7e7e7", lw=.6)
    fig.tight_layout()
    path = FIG / "glioma_apoptosis_source_descriptive_20261105.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def build_pages(summary: dict, plot: Path) -> list[list[dict]]:
    pages = copy.deepcopy(json.loads(BLOCKS.read_text(encoding="utf-8")))
    assert len(pages) == 16 and pages[10][0]["text"] == "References"
    assert pages[0][3]["text"] == "Abstract"
    pages[0][1]["text"] = (
        "Yao Zhao, Xiaoxuan Lu - Fudan University\n"
        "Yang Yang - Guangxi Medical University\n"
        "Di Zhu* - Suzhou Fusheng Pharmaceutical Technology Co., Ltd.\n"
        "*Correspondence: zhudifudan@163.com"
    )
    pages[0][2]["text"] = ""
    # Replace rather than append to keep the main paper within ten pages.
    pages[9][7]["text"] = (
        "The application path remains target intervention, cellular response, measured function and, eventually, tissue behavior. "
        "The newly supplied glioma assay files [12] contain drug-exposed apoptosis percentages and relative qPCR values, "
        "which we audit descriptively in Appendix F. They are not held-out FVcell predictions: the present model predicts "
        "terminal T-cell CRISPRi RNA responses, whereas these measurements concern a glioma small molecule. "
        "Drug-to-organ transfer still requires matched exposure and tissue data. The glioma study lacks pharmacokinetic and "
        "brain/tumor exposure measurements [12]; we do not infer blood-brain barrier transport or clinical benefit."
    )
    pages[9][10]["text"] = (
        "GSE278572 is evaluated from complete counts under the frozen protocol described above. "
        "The closer genome-scale CD4 resource at Zenodo 18876252 remains embargoed and contributes no labels. "
        "No public archival source-code release is documented for this draft."
    )
    pages[9][9]["text"] = pages[9][9]["text"].replace(
        "This is a working manuscript, not a claim that submission or public release has occurred.",
        "This draft is pending a documented public archival source-code release."
    )
    evidence_table = pages[15][1]
    assert evidence_table["rows"][1][0] == "Tumor-cell pharmacology"
    evidence_table["rows"][1][2] = "Audited apoptosis/qPCR observations; no FVcell compound-response prediction or glioma transfer validation"
    evidence_table["caption"] = (
        "Table S5. Published findings and supplied assay files remain attributed to their source study. "
        "The newly audited measurements do not become FVcell predictions."
    )
    assert pages[15][5]["kind"] == "paragraph"
    pages[15][5]["text"] = (
        "The received glioma subset includes Prism apoptosis percentages and relative qPCR entries, selected Ki67 gate PDFs "
        "and two plate-reader exports. Appendix F reports the extractable values and missing mappings. Complete RNA counts, "
        "independent experiment identifiers, plate maps, FCS files, rescue measurements and exposure data were not supplied. "
        "For the immune mechanism [11], matched per-sample wet-lab measurements remain unavailable."
    )
    pages[15][6]["text"] = pages[15][6]["text"].replace(
        "Before receiving outcomes,", "Before any predictive use of new outcomes,"
    )

    doses = [(0, "Control"), (2.5, "2.5 μM C37-28R"), (5, "5 μM C37-28R"), (10, "10 μM C37-28R")]
    apoptosis_rows = []
    for cell in ["U87-MG", "U251"]:
        row = [cell]
        for _, label in doses:
            r = summary[("Apoptosis.pzfx", cell, label)]
            row.append(f'{float(r["mean"]):.2f} ± {float(r["sample_sd"]):.2f} ({r["n_nonmissing_entries"]})')
        apoptosis_rows.append(row)
    pages.append([
        block("heading", "Appendix F  Supplied glioma assay files: descriptive audit"),
        block("paragraph", (
            "The data provider confirmed that the Prism label C37-28R refers to the publication's ZD37-28R [12]. "
            "Apoptosis.pzfx contains 24 nonmissing entries: three recorded values in each of four groups for U87-MG and U251. "
            "The source files do not identify biological replicate, batch or vehicle; reported n counts entries rather than confirmed "
            "independent experiments. No new significance test or efficacy estimate is inferred."
        )),
        {"kind": "figure", "path": str(plot), "caption": (
            "Figure S2. Descriptive Annexin V/PI apoptotic-cell percentages in the supplied Prism table. "
            "Markers are group means and bars are sample SD of three stored entries. "
            "The plot is a re-expression of source-study measurements, not an FVcell prediction."
        )},
        table(["Cell line", "Control", "2.5 μM", "5 μM", "10 μM"], apoptosis_rows,
              [68, 100, 100, 100, 100],
              "Table S6. Mean ± sample SD (nonmissing entries) from Apoptosis.pzfx; values are percentages. The publication's Figure 2e reports the same assay category."),
        block("paragraph", (
            "The recorded apoptosis means rise with dose in both lines (U87-MG: 7.93% control to 87.60% at 10 μM; "
            "U251: 6.77% to 93.17%). These are observations from the source study, with no paired genetic, transcriptomic "
            "or model-prediction record to score cross-modality transfer."
        )),
    ])

    genes = [
        ("PRRX1", "PRRX1"), ("PITX2", "PITX2"), ("SLUG (SNAI2)", "SLUG"),
        ("SREBP1 (SREBF1)", "SREBP1"),
    ]
    qpcr_rows = []
    for cell in ["U87", "U251"]:
        for shown, stored in genes:
            table_name = f"{stored}-{cell}"
            row = [f"{cell} / {shown}"]
            for _, label in doses:
                r = summary[("EMT-QPCR.pzfx", table_name, label)]
                row.append(f'{float(r["mean"]):.3f} ({r["n_nonmissing_entries"]})')
            qpcr_rows.append(row)
    pages.append([
        block("heading", "Appendix F  Relative qPCR and unresolved endpoints"),
        block("paragraph", (
            "EMT-QPCR.pzfx contains 167 nonmissing values across eight cell-line/gene tables and four concentrations. "
            "The stored values are relative expression measurements; no Cq records or sample-to-run map are supplied. "
            "SLUG denotes SNAI2 and SREBP1 denotes SREBF1 at the transcript level."
        )),
        table(["Cell line / transcript", "Control", "2.5 μM", "5 μM", "10 μM"], qpcr_rows,
              [124, 86, 86, 86, 86],
              "Table S7. Arithmetic mean (nonmissing entries) from EMT-QPCR.pzfx. Entries per group vary from four to six; sample SD and each value are retained in the companion CSV files. No claim of independent biological n or recomputed P values is made."),
        block("paragraph", (
            "At 10 μM, all eight stored mean qPCR values are lower than their respective controls, but several intermediate "
            "dose means are not monotone. The cellcycle.pzfx file contains 27 percentage entries without a named phase or "
            "outcome; it is extracted but not assigned to G1, S or G2/M. The two U251 plate files contain 104 numeric 450-nm "
            "wells without a well-to-treatment/dose map, blank controls or experiment identifiers, so no IC50 is refitted. "
            "The Ki67 PDFs contain selected gated plots rather than FCS events or a complete replicate table."
        )),
        block("paragraph", (
            "Source file SHA256 hashes, every stored Prism value, the descriptive summaries and unmapped plate wells accompany "
            "this draft. No provided source was overwritten. These assays may anchor a later, separately locked drug-response "
            "test if paired sample identifiers and appropriate model inputs become available."
        )),
    ])
    return pages


def write_md(pages: list[list[dict]]) -> Path:
    chunks = []
    for pg in pages:
        for b in pg:
            kind = b["kind"]
            if kind == "title": chunks.append("# " + b["text"])
            elif kind == "heading": chunks.append("## " + b["text"])
            elif kind in ("paragraph", "authors"): chunks.append(b["text"])
            elif kind == "equation": chunks.append("```text\n" + b["text"] + "\n```")
            elif kind == "figure": chunks.append(f'![{b["caption"]}](figures/{Path(b["path"]).name})')
            elif kind == "table":
                rows = [b["headers"], ["---"] * len(b["headers"]), *b["rows"]]
                chunks.append("\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows))
                if b.get("caption"): chunks.append(b["caption"])
        chunks.append("\n<!-- page break -->\n")
    path = WORK / "FVcell_Conference_Manuscript_20261105.md"
    path.write_text("\n\n".join(chunks), encoding="utf-8")
    return path


def render_pdf(pages: list[list[dict]]) -> Path:
    pdfmetrics.registerFont(TTFont("Body", "C:/Windows/Fonts/times.ttf"))
    pdfmetrics.registerFont(TTFont("BodyBold", "C:/Windows/Fonts/timesbd.ttf"))
    pdfmetrics.registerFont(TTFont("BodyItalic", "C:/Windows/Fonts/timesi.ttf"))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBold")
    styles = {
        "body": ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=12.3, spaceAfter=5.5),
        "heading": ParagraphStyle("head", fontName="BodyBold", fontSize=12, leading=14.5, spaceBefore=5, spaceAfter=6, keepWithNext=True),
        "title": ParagraphStyle("title", fontName="BodyBold", fontSize=17, leading=20, alignment=TA_CENTER, spaceAfter=10),
        "authors": ParagraphStyle("authors", fontName="Body", fontSize=11, leading=13, alignment=TA_CENTER, spaceAfter=7),
        "caption": ParagraphStyle("caption", fontName="BodyItalic", fontSize=10, leading=12, spaceAfter=6),
        "table": ParagraphStyle("cell", fontName="Body", fontSize=10, leading=11.6),
    }
    def para(value: str, style: str = "body") -> Paragraph:
        return Paragraph(html.escape(value).replace("\n", "<br/>"), styles[style])
    story = []
    for index, pg in enumerate(pages):
        if index: story.append(PageBreak())
        for b in pg:
            kind = b["kind"]
            if kind == "equation":
                formula = (r'$\hat{\delta}=\mu_{\mathrm{train}}+W_s h(u)+W_r[A(u)\odot\tanh(S(z)+b_c+v_d)]$'
                           if b["text"].startswith("delta_hat") else
                           r'$\mathrm{MAE}=\mathrm{mean}_d\ \mathrm{mean}_c\ \mathrm{mean}_a\ \mathrm{mean}_g\ |\delta_{dcag}-\hat{\delta}_{dcag}|$')
                eq_path = HERE / ("equation_" + str(len(b["text"])) + ".png")
                math_to_image(formula, str(eq_path), prop=FontProperties(size=11), dpi=300)
                arr = plt.imread(eq_path)
                width = min(468, arr.shape[1] * 72 / 300)
                story.extend([Spacer(1, 3), Image(str(eq_path), width=width, height=width * arr.shape[0] / arr.shape[1]), Spacer(1, 6)])
            elif kind in ("paragraph", "heading", "title", "authors"):
                story.append(para(b["text"], "body" if kind == "paragraph" else kind))
            elif kind == "figure":
                arr = plt.imread(b["path"])
                story.extend([Image(b["path"], width=468, height=468 * arr.shape[0] / arr.shape[1]), para(b["caption"], "caption")])
            elif kind == "table":
                data = [[para(str(v), "table") for v in row] for row in [b["headers"], *b["rows"]]]
                t = Table(data, colWidths=b.get("widths") or [468 / len(data[0])] * len(data[0]), repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eceff1")),
                    ("LINEABOVE", (0, 0), (-1, 0), .7, colors.black), ("LINEBELOW", (0, 0), (-1, 0), .5, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), .6, colors.black), ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.extend([t, Spacer(1, 5)])
                if b.get("caption"): story.append(para(b["caption"], "caption"))
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 10)
        canvas.drawRightString(540, 40, str(doc.page))
        canvas.restoreState()
    path = OUT / "FVcell_Conference_Manuscript_20261105.pdf"
    document = BaseDocTemplate(str(path), pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72,
                               bottomMargin=72, title=pages[0][0]["text"],
                               author="Yao Zhao; Xiaoxuan Lu; Yang Yang; Di Zhu")
    frame = Frame(72, 72, 468, 648, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    document.addPageTemplates(PageTemplate(id="paper", frames=frame, onPage=footer))
    document.build(story)
    return path


def latex_source(pages: list[list[dict]]) -> None:
    LATEX.mkdir(exist_ok=True)
    (LATEX / "images").mkdir(exist_ok=True)
    shutil.copy2(WORK / "latex" / "main.tex", LATEX / "main.tex")
    shutil.copy2(WORK / "references.bib", LATEX / "references.bib")
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\ensuremath{<}", ">": r"\ensuremath{>}"}
    def esc(value: object) -> str:
        return "".join(mapping.get(ch, ch) for ch in str(value)).replace("≤", r"\ensuremath{\leq}").replace("±", r"\ensuremath{\pm}").replace("μ", r"\ensuremath{\mu}")
    tex = []
    for index, pg in enumerate(pages):
        if index: tex.append(r"\clearpage")
        for b in pg:
            kind = b["kind"]
            value = b.get("text", "")
            if kind == "title": tex.append(r"\begin{center}{\Large\bfseries " + esc(value) + r"}\end{center}")
            elif kind == "authors": tex.append(r"\begin{center}" + esc(value).replace("\n", r"\\") + r"\end{center}")
            elif kind == "heading": tex.append(r"\subsection*{" + esc(value) + "}")
            elif kind == "paragraph": tex.append(esc(value) + "\n")
            elif kind == "equation":
                if value.startswith("delta_hat"):
                    tex.append(r"\[\widehat{\delta}=\mu_{\mathrm{train}}+W_s h(u)+W_r\left[A(u)\odot\tanh\{S(z)+b_c+v_d\}\right].\]")
                else:
                    tex.append(r"\[\mathrm{MAE}=\underset{d}{\mathrm{mean}}\ \underset{c}{\mathrm{mean}}\ \underset{a}{\mathrm{mean}}\ \underset{g}{\mathrm{mean}}\ |\delta_{dcag}-\widehat{\delta}_{dcag}|.\]")
            elif kind == "figure":
                path = Path(b["path"])
                shutil.copy2(path, LATEX / "images" / path.name)
                tex.append(r"\begin{center}\includegraphics[width=\linewidth]{images/" + path.name + r"}\end{center}" + "\n" + esc(b["caption"]))
            elif kind == "table":
                widths = b.get("widths") or [468 / len(b["headers"])] * len(b["headers"])
                spec = "@{}" + " ".join(r">{\raggedright\arraybackslash}p{" + f"{w/468*.93:.3f}" + r"\linewidth}" for w in widths) + "@{}"
                tex.append(r"\begin{center}\begin{tabular}{" + spec + r"}\toprule")
                tex.append(" & ".join(esc(v) for v in b["headers"]) + r"\\\midrule")
                for row in b["rows"]: tex.append(" & ".join(esc(v) for v in row) + r"\\")
                tex.append(r"\bottomrule\end{tabular}\end{center}")
                if b.get("caption"): tex.append(esc(b["caption"]))
    (LATEX / "body.tex").write_text("\n".join(tex), encoding="utf-8")


def main() -> None:
    source_manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    assert all(sha256(ROOT / "data_sources" / name) == rec["sha256"] for name, rec in source_manifest["sources"].items())
    summary = read_summary()
    plot = make_plot(summary)
    pages = build_pages(summary, plot)
    assert pages[0][4]["text"] == json.loads(BLOCKS.read_text(encoding="utf-8"))[0][4]["text"]
    blocks_path = WORK / "manuscript_blocks_20261105.json"
    blocks_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = write_md(pages)
    pdf = render_pdf(pages)
    latex_source(pages)
    metadata = {
        "ordered_authors": [
            {"name": "Yao Zhao", "affiliation": "Fudan University", "email": "2158027559@qq.com"},
            {"name": "Xiaoxuan Lu", "affiliation": "Fudan University", "email": None},
            {"name": "Yang Yang", "affiliation": "Guangxi Medical University", "email": None},
            {"name": "Di Zhu", "affiliation": "Suzhou Fusheng Pharmaceutical Technology Co., Ltd.", "email": "zhudifudan@163.com", "corresponding": True},
        ],
        "abstract_status": "User reports registered; exact registered text/receipt not independently inspected.",
        "missing_author_emails": ["Xiaoxuan Lu", "Yang Yang"],
    }
    (WORK / "AUTHOR_METADATA_20261105.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    reader = PdfReader(str(pdf))
    page_text = [p.extract_text() or "" for p in reader.pages]
    ref_page = next(i for i, text in enumerate(page_text, 1) if re.search(r"\bReferences\b", text) and i > 8)
    assert ref_page - 1 <= 10, f"Main body exceeds 10 pages: {ref_page - 1}"
    assert len(reader.pages) == len(pages), f"Unexpected page spill: {len(reader.pages)} vs {len(pages)}"
    assert "Appendix F" in page_text[-2] and "Appendix F" in page_text[-1]
    qa = {"status": "built_pending_visual_review", "pages": len(reader.pages), "main_pages": ref_page - 1,
          "source_count": len(source_manifest["sources"]), "source_hashes_unchanged": True,
          "abstract_unchanged": True, "new_glioma_measurements_are_descriptive_only": True,
          "pdf_sha256": sha256(pdf), "markdown_sha256": sha256(markdown), "blocks_sha256": sha256(blocks_path)}
    (HERE / "manuscript_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
