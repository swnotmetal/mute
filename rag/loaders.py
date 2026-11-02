"""Multi-format document loader: txt/md/pdf/docx/xlsx (png is reported, not OCR'd)."""

from dataclasses import dataclass
from pathlib import Path

SUPPORTED = (".txt", ".md", ".pdf", ".docx", ".xlsx")
# .xls (legacy binary) needs xlrd, not openpyxl -> reported as unsupported.
UNSUPPORTED_HINTS = {".png": "image OCR not enabled", ".xls": "legacy .xls not supported"}


@dataclass
class Document:
    text: str
    source: str


def load_dir(root: str | Path, exts: tuple[str, ...] = SUPPORTED):
    """Recursively load supported files under `root`.

    Returns (documents, skipped) where skipped = [(path, reason), ...].
    """
    root = Path(root)
    docs: list[Document] = []
    skipped: list[tuple[str, str]] = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in UNSUPPORTED_HINTS:
            skipped.append((str(p), UNSUPPORTED_HINTS[suf]))
            continue
        if suf not in exts:
            continue
        try:
            text = _load_one(p)
        except Exception as e:  # one bad file must not kill the whole index
            skipped.append((str(p), f"{type(e).__name__}: {e}"))
            continue
        if text and text.strip():
            docs.append(Document(text=text, source=str(p)))
        else:
            skipped.append((str(p), "empty text"))

    return docs, skipped


def _load_one(p: Path) -> str:
    suf = p.suffix.lower()

    if suf in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")

    if suf == ".pdf":
        from pypdf import PdfReader
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages)

    if suf == ".docx":
        from docx import Document as Docx
        return "\n".join(par.text for par in Docx(str(p)).paragraphs)

    if suf == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
        lines: list[str] = []
        for ws in wb.worksheets:
            lines.append(f"[sheet: {ws.title}]")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)

    return ""
