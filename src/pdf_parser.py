import re
import pymupdf
import requests

MAX_PAGES = 60
MAX_MB = 30
MIN_CHARS = 2000

HEADINGS = ["abstract", "introduction", "related work", "background", "preliminaries",
            "method", "methods", "methodology", "approach", "experiments", "experimental setup",
            "results", "evaluation", "discussion", "limitations", "conclusion",
            "conclusions", "references"]
HEAD_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?(" + "|".join(HEADINGS) + r")\b[^\n]{0,40}$",
    re.I | re.M)


def download_pdf(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    if len(r.content) > MAX_MB * 1024 * 1024:
        raise ValueError("PDF too large")
    return r.content


def extract_text(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    n = min(len(doc), MAX_PAGES)
    return "\n".join(doc[i].get_text("text") for i in range(n))


def split_sections(text: str) -> dict:
    """Split by common headings. References are kept separately."""
    matches = list(HEAD_RE.finditer(text))
    if not matches:
        return {"body": text}
    sections = {}
    if matches[0].start() > 0:
        sections["front_matter"] = text[:matches[0].start()]
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = m.group(1).lower()
        sections[name] = sections.get(name, "") + "\n" + text[m.end():end]
    return sections


def parse_pdf(url: str) -> tuple[dict, str]:
    """Returns (sections, quality). Raises on download errors."""
    text = extract_text(download_pdf(url))
    if len(text.strip()) < MIN_CHARS:
        return {}, "abstract_only"
    return split_sections(text), "full"