"""Shared generation guidance and deterministic, in-memory PDF layout rewards."""
from __future__ import annotations

import os
import re
from pathlib import Path

from fpdf import FPDF

# Personal header links from the supplied sample, including embedded hyperlinks.
HEADER_LINKS = (
    "https://www.linkedin.com/in/kelvin-mock-548666173/",
    "https://github.com/kmock930",
    "https://github.coventry.ac.uk/mockk",
    "https://github.com/KelvinMock",
    "https://shorturl.at/38Rly",
    "https://devpost.com/kmock930",
)
SECTION_NAMES = {
    "summary", "professional summary", "skills", "technical skills", "education",
    "experience", "work experience", "professional experience", "projects",
    "certifications", "achievements", "publications", "volunteering", "references",
}


def conciseness_prompt(task_type: str) -> str:
    limit = 2 if task_type == "resume" else 1
    return (
        f"Conciseness: Keep the complete document within {limit} page(s), including "
        "the header. Use Times New Roman, 10 pt body text and 12 pt section headings. "
        "Return plain text with each section heading on its own line, and preserve "
        "paragraph breaks. Edit repetition rather than shrinking the font. "
        "Keep all header lines together before the first blank line. "
        "Begin with a header containing Kelvin Mock, Ottawa, ON (willing to relocate), "
        "+1 (437) 328-1046, kelvin.mock@gmail.com, and all these personal links: "
        + ", ".join(HEADER_LINKS)
        + ". Preserve any additional applicant-provided header links as full URLs; "
        "retain relevant project links with their projects."
    )


def _font_paths() -> tuple[Path, Path]:
    # Workers must install the actual font; a metric-incompatible fallback would
    # silently change the reward. Do not redistribute proprietary font files.
    directory = Path(os.environ.get("TIMES_NEW_ROMAN_FONT_DIR", "C:/Windows/Fonts"))
    paths = directory / "times.ttf", directory / "timesbd.ttf"
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(
            "Install Times New Roman and set TIMES_NEW_ROMAN_FONT_DIR to the "
            "directory containing times.ttf and timesbd.ttf on every reward worker."
        )
    return paths


def document_page_count(text: str) -> int:
    """Render all text to a PDF bytearray, never a file, and count its pages.

    Scoring uses Letter, 0.75-inch margins and 1.15 line spacing. Headings
    are standalone Markdown headings, known section titles, or short uppercase
    lines. Blank lines and explicit form-feed page breaks consume space.
    """
    regular, bold = _font_paths()
    pdf = FPDF(format="Letter", unit="pt")
    pdf.set_margins(54, 54, 54)
    pdf.set_auto_page_break(True, margin=54)
    pdf.add_font("Times New Roman", fname=str(regular))
    pdf.add_font("Times New Roman", style="B", fname=str(bold))
    pdf.add_page()
    for index, block in enumerate(text.split("\f")):
        if index:
            pdf.add_page()
        for line in block.splitlines():
            clean = re.sub(r"^\s*#{1,6}\s+", "", line).strip()
            clean = clean.removeprefix("**").removesuffix("**")
            heading = bool(clean) and (
                line.lstrip().startswith("#")
                or clean.lower().rstrip(":") in SECTION_NAMES
                or (clean.isupper() and len(clean) < 70)
            )
            size = 12 if heading else 10
            pdf.set_font("Times New Roman", "B" if heading else "", size)
            if not clean:
                pdf.ln(11.5)
                continue
            pdf.multi_cell(0, size * 1.15, clean, new_x="LMARGIN", new_y="NEXT")
    pdf.output()  # fpdf2 returns bytes in memory when no filename is supplied.
    return pdf.page_no()


def layout_rewards(text: str, task_type: str, source: str = "") -> dict[str, float]:
    pages = document_page_count(text)
    limit = 2 if task_type == "resume" else 1
    # Overflow never earns page-fit credit; the existing length signal remains.
    page_fit = float(bool(text.strip()) and pages <= limit)
    header = text.split("\n\n", 1)[0].lower()
    required = ["Kelvin Mock", "Ottawa", "kelvin.mock@gmail.com", *HEADER_LINKS]
    # Additional source-header URLs must also survive generation.
    required.extend(re.findall(r"https?://[^\s<>]+", source.split("\n\n", 1)[0]))
    header_reward = sum(item.lower().rstrip("/") in header for item in required) / len(required)
    return {"page_count": float(pages), "page_fit_reward": page_fit, "header_reward": header_reward}
