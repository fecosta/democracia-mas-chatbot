from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pypdf import PdfReader

# Reduce pypdf chatter like "Ignoring wrong pointing object ..."
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)


@dataclass
class Section:
    path: str
    level: int
    page_start: int
    page_end: int
    text: str


@dataclass
class ExtractionReport:
    page_count: int
    extracted_pages: int
    total_chars: int
    empty_pages: int
    warnings: List[str]


# Bullet / heading patterns
_BULLET_RE = re.compile(r"^\s*(?:[\-\*\u2022\u2013\u2014\u00b7]|•|–|—)\s+")
_ENUM_BULLET_RE = re.compile(r"^\s*(?:\d{1,3}\.|[A-Z]\)|\([a-zA-Z]\))\s+")
_NUM_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,6})\s+(.{2,200})\s*$")

# A “heading-ish” line tends to be short and not end with punctuation.
_HEADING_END_PUNCT = re.compile(r"[.!?;:,]\s*$")


def extract_text_by_page(pdf_path: str) -> Tuple[List[str], ExtractionReport]:
    """
    Robust page-by-page extraction.
    Returns (raw_pages, report). raw_pages retains original newlines from pypdf.
    """
    warnings: List[str] = []
    pages: List[str] = []

    try:
        reader = PdfReader(pdf_path, strict=False)
    except Exception as e:
        # total failure
        return [], ExtractionReport(
            page_count=0, extracted_pages=0, total_chars=0, empty_pages=0,
            warnings=[f"Failed to open PDF: {e}"],
        )

    page_count = len(reader.pages)
    extracted_pages = 0
    empty_pages = 0
    total_chars = 0

    for i, p in enumerate(reader.pages, start=1):
        try:
            txt = p.extract_text() or ""
        except Exception as e:
            txt = ""
            warnings.append(f"Page {i}: extract_text failed: {e}")

        pages.append(txt)

        if txt.strip():
            extracted_pages += 1
            total_chars += len(txt)
        else:
            empty_pages += 1

    return pages, ExtractionReport(
        page_count=page_count,
        extracted_pages=extracted_pages,
        total_chars=total_chars,
        empty_pages=empty_pages,
        warnings=warnings,
    )


def _fix_hyphenation_keep_lines(t: str) -> str:
    # demo-\ncracy -> democracy (keep paragraph breaks)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    return t


def _normalize_preserve_lines(t: str) -> str:
    """
    Normalize text but preserve line structure for heading/bullet detection.
    - Keep newlines
    - Collapse weird whitespace
    - Preserve blank lines (paragraph boundaries)
    """
    t = t.replace("\x00", "")
    t = unicodedata.normalize("NFKC", t)
    t = _fix_hyphenation_keep_lines(t)

    # Normalize line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Trim trailing spaces per line
    t = "\n".join([line.strip() for line in t.split("\n")])

    # Collapse internal spaces
    t = re.sub(r"[ \t\u00a0]+", " ", t)

    # Collapse too many blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def _strip_repeated_headers_footers(raw_pages: List[str], n_lines: int = 2) -> List[str]:
    """
    Improved heuristic: look at first/last N non-empty lines per page and remove the most frequent
    header/footer blocks if they appear on >= 40% of pages.
    """
    def top_block(lines: List[str], take_first: bool) -> Optional[str]:
        # Build a canonical block string of first/last n_lines non-empty lines
        cleaned = [l.strip() for l in lines if l.strip()]
        if not cleaned:
            return None
        block_lines = cleaned[:n_lines] if take_first else cleaned[-n_lines:]
        # Avoid removing tiny blocks
        block = " | ".join(block_lines).strip()
        if len(block) < 6:
            return None
        return block

    first_blocks: List[str] = []
    last_blocks: List[str] = []

    per_page_first: List[Optional[str]] = []
    per_page_last: List[Optional[str]] = []

    for p in raw_pages:
        lines = p.splitlines()
        fb = top_block(lines, take_first=True)
        lb = top_block(lines, take_first=False)
        per_page_first.append(fb)
        per_page_last.append(lb)
        if fb:
            first_blocks.append(fb)
        if lb:
            last_blocks.append(lb)

    def most_common(blocks: List[str]) -> Tuple[Optional[str], int]:
        if not blocks:
            return None, 0
        freq: dict[str, int] = {}
        for b in blocks:
            freq[b] = freq.get(b, 0) + 1
        best = max(freq.items(), key=lambda kv: kv[1])
        return best[0], best[1]

    first, fcount = most_common(first_blocks)
    last, lcount = most_common(last_blocks)

    n_pages = max(1, len(raw_pages))
    remove_first = first if first and (fcount / n_pages) >= 0.4 else None
    remove_last = last if last and (lcount / n_pages) >= 0.4 else None

    cleaned_pages: List[str] = []

    for p in raw_pages:
        lines = [l.rstrip() for l in p.splitlines()]
        non_empty = [l.strip() for l in lines if l.strip()]

        # Determine actual first/last block on this page
        fb = top_block(lines, take_first=True)
        lb = top_block(lines, take_first=False)

        # Remove first block lines if match
        out_lines = lines[:]
        if remove_first and fb == remove_first:
            # remove first n_lines non-empty lines (preserving intervening empties)
            removed = 0
            new_out = []
            for line in out_lines:
                if removed < n_lines and line.strip():
                    removed += 1
                    continue
                new_out.append(line)
            out_lines = new_out

        # Remove last block lines if match
        if remove_last and lb == remove_last:
            removed = 0
            new_out = []
            # remove last n_lines non-empty lines from the end
            for line in reversed(out_lines):
                if removed < n_lines and line.strip():
                    removed += 1
                    continue
                new_out.append(line)
            out_lines = list(reversed(new_out))

        cleaned_pages.append("\n".join(out_lines))

    return cleaned_pages


def _looks_like_heading(line: str) -> Tuple[bool, Optional[str], int]:
    """
    Returns (is_heading, normalized_title, level)
    """
    s = line.strip()
    if not s:
        return False, None, 0

    # Numbered heading
    m = _NUM_HEADING_RE.match(s)
    if m:
        num, title = m.group(1), m.group(2).strip()
        level = min(6, num.count(".") + 1)
        if len(title) >= 2:
            return True, f"{num} {title}", level

    # ALL CAPS heading
    alpha = sum(c.isalpha() for c in s)
    is_caps = len(s) <= 90 and s.isupper() and alpha >= 6
    if is_caps and not _HEADING_END_PUNCT.search(s):
        return True, s.title(), 2

    # Title-ish short line (heuristic)
    if 4 <= len(s) <= 70 and not _HEADING_END_PUNCT.search(s):
        # not a bullet
        if not (_BULLET_RE.match(s) or _ENUM_BULLET_RE.match(s)):
            # avoid lines that are obviously sentences
            words = s.split()
            if 1 <= len(words) <= 10:
                return True, s, 2

    return False, None, 0


def _is_bullet(line: str) -> Tuple[bool, str]:
    s = line.strip()
    if not s:
        return False, ""
    if _BULLET_RE.match(s):
        return True, _BULLET_RE.sub("", s).strip()
    if _ENUM_BULLET_RE.match(s):
        return True, _ENUM_BULLET_RE.sub("", s).strip()
    return False, ""


def build_sections_from_pdf(pdf_path: str, filename: str) -> List[Section]:
    raw_pages, _report = extract_text_by_page(pdf_path)

    # Remove repeated headers/footers before normalization
    raw_pages = _strip_repeated_headers_footers(raw_pages, n_lines=2)

    # Normalize but keep line breaks
    pages = [_normalize_preserve_lines(p) for p in raw_pages]

    # Build a stream of (page_no, line)
    all_lines: List[Tuple[int, str]] = []
    for page_no, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for line in page_text.split("\n"):
            # Keep blank lines as separators (we'll use them to separate paragraphs)
            all_lines.append((page_no, line))

    if not all_lines:
        return []

    sections: List[Section] = []
    current_title = f"{filename}"
    current_level = 1
    current_start = all_lines[0][0]
    buf: List[str] = []

    def flush(end_page: int) -> None:
        nonlocal buf, current_title, current_level, current_start
        text = "\n".join(buf).strip()

        # Clean extra blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if text:
            sections.append(
                Section(
                    path=current_title,
                    level=current_level,
                    page_start=current_start,
                    page_end=end_page,
                    text=text,
                )
            )
        buf = []

    for page_no, line in all_lines:
        s = line.strip()

        # treat blank line as paragraph separator
        if not s:
            # only keep one blank line
            if buf and buf[-1] != "":
                buf.append("")
            continue

        # headings
        is_head, head_title, head_level = _looks_like_heading(s)
        if is_head and head_title:
            # start new section
            flush(page_no)
            current_title = f"{filename} > {head_title}"
            current_level = head_level
            current_start = page_no
            continue

        # bullets
        is_bul, bul_text = _is_bullet(s)
        if is_bul and bul_text:
            buf.append(f"• {bul_text}")
        else:
            buf.append(s)

    flush(all_lines[-1][0])
    return sections