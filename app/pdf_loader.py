"""
pdf_loader.py (PRODUCTION VERSION)

Upgrades:
- Structured semantic tagging for RAG
- Markdown table formatting
- Metadata injection (DOCTOR, PROCEDURE, DATA_TYPE)
- Currency labeling
- Deterministic graft & pricing extraction
- Version-aware caching
"""

import pdfplumber
import re
import os

PDF_PATH = "app/data/hair_transplant.pdf"
MD_PATH  = "app/data/hair_transplant.md"


# ─────────────────────────────────────────────
# Basic Cleaning Helpers
# ─────────────────────────────────────────────

def clean_line(line: str) -> str:
    return re.sub(r'\s+', ' ', line.strip())


def is_page_header(line: str) -> bool:
    return "HairRevive Clinic — Confidential" in line


def is_section_heading(line: str) -> bool:
    return bool(re.match(r'^\d+\.\s+[A-Z]', line))


def is_subsection_heading(line: str) -> bool:
    return bool(re.match(r'^\d+\.\d+\s+[A-Z]', line))


def is_faq_question(line: str) -> bool:
    return line.startswith("Q:")


def is_bullet(line: str) -> bool:
    return line.startswith("•")


# ─────────────────────────────────────────────
# Table Detection
# ─────────────────────────────────────────────

TABLE_KEYWORDS = [
    "Day Opening Time",
    "Graft Range Price",
    "Procedure Description Price",
    "Stage Description Grafts",
    "Country Average Cost",
    "Package Name Includes",
    "Treatment Description Price",
]


def is_table_header_row(line: str) -> bool:
    return any(keyword in line for keyword in TABLE_KEYWORDS)


def is_numeric_row(line: str) -> bool:
    return bool(re.search(r'\d', line)) and len(line.split()) >= 2


# ─────────────────────────────────────────────
# PDF Extraction
# ─────────────────────────────────────────────

def extract_all_text(pdf_path: str) -> list[str]:
    all_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or ""
            for line in text.splitlines():
                all_lines.append(line)
    return all_lines


# ─────────────────────────────────────────────
# STRUCTURED MARKDOWN CONVERSION
# ─────────────────────────────────────────────

def lines_to_markdown(lines: list[str]) -> str:

    output = []
    in_table = False

    for raw_line in lines:
        line = clean_line(raw_line)

        if not line:
            if output and output[-1] != "":
                output.append("")
            continue

        if is_page_header(line):
            continue

        # ───────────── SECTION HEADINGS ─────────────

        if is_section_heading(line):
            output.append(f"\n## SECTION: {line}")
            continue

        if is_subsection_heading(line):
            output.append(f"\n### SUBSECTION: {line}")
            continue

        # ───────────── FAQ ─────────────

        if is_faq_question(line):
            output.append(f"\n[DATA_TYPE: FAQ]")
            output.append(f"**{line}**")
            continue

        # ───────────── BULLETS ─────────────

        if is_bullet(line):
            output.append(f"- {line[1:].strip()}")
            continue

        # ───────────── TABLE HEADER ─────────────

        if is_table_header_row(line):
            in_table = True
            output.append("\n[DATA_TYPE: TABLE]")
            columns = re.split(r'\s{2,}', line)
            output.append("| " + " | ".join(columns) + " |")
            output.append("|" + " --- |" * len(columns))
            continue

        # ───────────── TABLE ROWS ─────────────

        if in_table and is_numeric_row(line):
            columns = re.split(r'\s{2,}', line)
            output.append("| " + " | ".join(columns) + " |")
            continue

        if in_table and not is_numeric_row(line):
            in_table = False

        # ───────────── DOCTOR TAGGING ─────────────

        doctor_match = re.match(r'(Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)', line)
        if doctor_match:
            doctor_name = doctor_match.group(1)
            output.append(f"\n[DATA_TYPE: DOCTOR_PROFILE]")
            output.append(f"[DOCTOR: {doctor_name}]")
            output.append(line)
            continue

        # ───────────── STAGE → GRAFT EXTRACTION ─────────────

        stage_match = re.match(r'Stage\s*(\d+).*?(\d{3,4})\s*-\s*(\d{3,4})', line)
        if stage_match:
            stage = stage_match.group(1)
            graft_min = stage_match.group(2)
            graft_max = stage_match.group(3)

            output.append("\n[DATA_TYPE: GRAFT_REQUIREMENT]")
            output.append(f"[STAGE: {stage}]")
            output.append(f"[GRAFT_MIN: {graft_min}]")
            output.append(f"[GRAFT_MAX: {graft_max}]")
            output.append(line)
            continue

        # ───────────── CURRENCY DETECTION ─────────────

        if "PKR" in line:
            output.append("[CURRENCY: PKR]")
        if "$" in line or "USD" in line:
            output.append("[CURRENCY: USD]")

        output.append(line)

    markdown = "\n".join(output)

    # Normalize spacing
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    markdown = re.sub(r'[ \t]+$', '', markdown, flags=re.MULTILINE)

    return markdown.strip()


# ─────────────────────────────────────────────
# Main Conversion
# ─────────────────────────────────────────────

def pdf_to_markdown(pdf_path: str) -> str:
    lines = extract_all_text(pdf_path)
    return lines_to_markdown(lines)


def load_or_convert(pdf_path: str = PDF_PATH, md_path: str = MD_PATH) -> str:

    # Version-aware caching
    if os.path.exists(md_path) and \
       os.path.getmtime(md_path) > os.path.getmtime(pdf_path):
        print(f"[PDFLoader] Loading cached markdown: {md_path}")
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"[PDFLoader] Converting PDF to structured Markdown...")
    markdown = pdf_to_markdown(pdf_path)

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"[PDFLoader] Saved structured markdown: {md_path}")
    return markdown


# ─────────────────────────────────────────────
# CLI Preview
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    md  = sys.argv[2] if len(sys.argv) > 2 else MD_PATH

    result = load_or_convert(pdf, md)

    print("\n" + "=" * 70)
    print("MARKDOWN PREVIEW (first 3000 chars)")
    print("=" * 70)
    print(result[:3000])
    print("=" * 70)
    print(f"\nTotal characters : {len(result)}")
    print(f"Total lines      : {result.count(chr(10))}")
    print(f"Saved to         : {md}")