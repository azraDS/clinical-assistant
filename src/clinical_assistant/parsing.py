"""Parse the KDIGO guideline PDF into statement units (recommendations + practice points) with
full heading paths, section rationale, and captioned tables/figures. See notebooks/README.md §1.

Ported from notebooks/02_retrieval_pipeline.ipynb §1-4. The docling parse itself is cached to
disk (config.DOCLING_CACHE); this module is safe to import without docling ever touching the PDF
if that cache already exists.
"""

import re
from collections import Counter, defaultdict

from . import config
from .text_clean import clean_text

STATEMENT_RE = re.compile(r"^(Recommendation|Practice Point)\s+(\d+(?:\.\d+){1,4})\s*:\s*")
NUMBERED_HEADER_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+\S")
CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\s*[:.]\s*(.+)")
GRADE_RE = re.compile(r"\((\d[A-D])\)\s*\.?\s*$")
TABLE_REF_RE = re.compile(r"\b(Table|Figure)\s+(\d+)\b")


def _load_body_items():
    from docling_core.types.doc import DoclingDocument, PictureItem, SectionHeaderItem, TableItem

    if config.DOCLING_CACHE.exists():
        doc = DoclingDocument.load_from_json(config.DOCLING_CACHE)
    else:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        doc = converter.convert(config.GUIDELINE_PATH).document
        config.DOCLING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        doc.save_as_json(config.DOCLING_CACHE)

    items = [
        item for item, _level in doc.iterate_items() if str(getattr(item, "content_layer", "")) == "ContentLayer.BODY"
    ]
    return doc, items, SectionHeaderItem, TableItem, PictureItem


def _find_header_index(items, SectionHeaderItem, pattern, start=0):
    for i in range(start, len(items)):
        it = items[i]
        if isinstance(it, SectionHeaderItem) and re.match(pattern, it.text.strip()):
            return i
    raise ValueError(pattern)


def _is_real_subheading(text: str) -> bool:
    # docling also labels figure labels ("K + >5.5 mmol/l"), URL fragments, single letters and
    # misplaced statement lines as headers; those must not become part of heading_path.
    return (
        len(text) >= 4
        and not re.search(r"[\d/:]", text)
        and text[0].isupper()
        and not re.match(r"^(Practice Point|Recommendation|Rationale\b)", text)
    )


def _page_of(item):
    return item.prov[0].page_no if item.prov else None


def _walk_statements(items, start, end, source, SectionHeaderItem, TableItem, PictureItem, section_titles):
    statements = []
    section_prose = defaultdict(list)  # leaf section id -> rationale paragraphs
    chapter = section = None
    section_title = None
    subheading = None
    current = None

    for item in items[start:end]:
        if isinstance(item, (TableItem, PictureItem)):
            continue
        text = clean_text(getattr(item, "text", "") or "")
        if not text:
            continue

        # Checked before the item type: docling sometimes labels a statement as a
        # ListItem or SectionHeaderItem (e.g. PP 2.2.5, PP 3.3.1.3)
        m = STATEMENT_RE.match(text)
        if m:
            kind = "recommendation" if m.group(1) == "Recommendation" else "practice_point"
            grade = GRADE_RE.search(text)
            current = {
                "statement_id": f"{'Rec' if kind == 'recommendation' else 'PP'} {m.group(2)}",
                "number": m.group(2),
                "type": kind,
                "grade": grade.group(1) if grade else None,
                "chapter": chapter,
                "section": section,
                "section_title": section_title,
                "subheading": subheading,
                "text": text,
                "page": _page_of(item),
                "source": source,
            }
            statements.append(current)
            continue

        if isinstance(item, SectionHeaderItem):
            current = None
            m_ch = CHAPTER_RE.match(text)
            m_num = NUMBERED_HEADER_RE.match(text)
            if m_ch:
                chapter = f"Chapter {m_ch.group(1)}: {m_ch.group(2)}"
                section = section_title = subheading = None
            elif m_num:
                section, section_title, subheading = m_num.group(1), text, None
                section_titles[m_num.group(1)] = text
            elif not re.match(r"^(Table|Figure)\s+\d+", text) and _is_real_subheading(text):
                subheading = text.rstrip(".")
            continue

        from docling_core.types.doc import ListItem

        if isinstance(item, ListItem) and current is not None:
            current["text"] += "\n" + text
            continue

        # any other prose: rationale for the leaf section; ends list attachment
        current = None
        if section:
            section_prose[section].append(text)

    return statements, section_prose


def _dedupe(summary, body):
    by_id = {}
    for s in body:
        # a statement can legitimately be repeated inside the body (e.g. restated in a figure);
        # keep the first body occurrence
        by_id.setdefault(s["statement_id"], dict(s))
    summary_by_id = {}
    for s in summary:
        summary_by_id.setdefault(s["statement_id"], s)

    for sid, s in summary_by_id.items():
        if sid not in by_id:
            by_id[sid] = dict(s)
        elif len(s["text"]) > len(by_id[sid]["text"]) + 20:
            by_id[sid]["text"] = s["text"]
            by_id[sid]["text_from_summary"] = True

    def sort_key(sid):
        kind, num = sid.split(" ")
        return [int(x) for x in num.split(".")] + [0 if kind == "Rec" else 1]

    return [by_id[k] for k in sorted(by_id, key=sort_key)]


def parse_statements() -> tuple[list[dict], dict]:
    """Parse the guideline into statement units + captioned tables/figures.

    Returns (statements, table_units). Each statement has heading_path built from every
    numbered ancestor plus a trailing subheading, section_rationale, table_refs and domain.
    """
    doc, items, SectionHeaderItem, TableItem, PictureItem = _load_body_items()

    summary_start = _find_header_index(items, SectionHeaderItem, r"Summary of recommendation statements")
    body_start = _find_header_index(items, SectionHeaderItem, r"Chapter 1:", summary_start + 2)
    body_end = _find_header_index(items, SectionHeaderItem, r"Chapter 6:", body_start)  # ch.6 = research recs, excluded

    section_titles: dict[str, str] = {}  # "3.4" -> "3.4 Blood pressure control"
    summary_statements, _ = _walk_statements(
        items, summary_start, body_start, "summary", SectionHeaderItem, TableItem, PictureItem, section_titles
    )
    body_statements, section_prose = _walk_statements(
        items, body_start, body_end, "body", SectionHeaderItem, TableItem, PictureItem, section_titles
    )

    statements = _dedupe(summary_statements, body_statements)

    for s in statements:
        if s["type"] == "recommendation" and not s["grade"]:
            m = re.search(r"\((\d[A-D])\)", s["text"])
            s["grade"] = m.group(1) if m else None
        s["section_rationale"] = "\n".join(section_prose.get(s["section"], []))
        # every numbered ancestor, e.g. "3.15 Cardiovascular disease..." AND "3.15.1 Lipid management"
        parts = s["section"].split(".") if s["section"] else []
        numbered = [
            section_titles[p] for p in (".".join(parts[:n]) for n in range(2, len(parts) + 1)) if p in section_titles
        ]
        s["heading_path"] = " › ".join(x for x in [s["chapter"], *numbered, s["subheading"]] if x)
        s["table_refs"] = sorted({f"{k} {n}" for k, n in TABLE_REF_RE.findall(s["text"])})
        # care domain = the 2-level section the statement belongs to, e.g. "3.4 Blood pressure control"
        s["domain"] = section_titles.get(".".join(s["number"].split(".")[:2]), s["chapter"])

    table_units = _parse_tables(doc, items, body_end, TableItem, PictureItem)
    _check_completeness(statements)
    return statements, table_units


def _parse_tables(doc, items, body_end, TableItem, PictureItem) -> dict:
    table_units = {}
    for item in items[:body_end]:
        if not isinstance(item, TableItem):
            continue
        caption = clean_text(item.caption_text(doc) or "")
        m = re.match(r"^(Table|Figure)\s*(\d+)\s*\|?\s*(.*)", caption)
        if not m:
            continue
        key = f"{m.group(1)} {m.group(2)}"
        markdown = clean_text(item.export_to_markdown(doc))
        # keep the largest rendering if a table is repeated / continued
        if key not in table_units or len(markdown) > len(table_units[key]["markdown"]):
            table_units[key] = {
                "table_id": key,
                "caption": caption,
                "markdown": markdown,
                "page": _page_of(item),
            }

    # Figures that are images: keep caption only (e.g. Figure 55, heat maps)
    for item in items[:body_end]:
        if isinstance(item, PictureItem):
            caption = clean_text(item.caption_text(doc) or "")
            m = re.match(r"^(Figure)\s*(\d+)\s*\|?\s*(.*)", caption)
            if m and f"Figure {m.group(2)}" not in table_units:
                key = f"Figure {m.group(2)}"
                table_units[key] = {"table_id": key, "caption": caption, "markdown": "", "page": _page_of(item)}
    return table_units


def _check_completeness(statements: list[dict]) -> None:
    """Cross-check against an independent pypdf extraction (whole document, chapters 1-5)."""
    from .text_clean import pypdf_reference

    ref_text = pypdf_reference.full_text
    pypdf_ids = {
        f"{'Rec' if kind == 'Recommendation' else 'PP'} {num}"
        for kind, num in re.findall(r"(Recommendation|Practice Point)\s+(\d+(?:\.\d+){1,4})\s*:", ref_text)
        if num.split(".")[0] in {"1", "2", "3", "4", "5"}
    }
    ours = {s["statement_id"] for s in statements}
    if pypdf_ids != ours:
        raise AssertionError(
            f"statement completeness check failed — missing: {sorted(pypdf_ids - ours)}, "
            f"extra: {sorted(ours - pypdf_ids)}"
        )


def statement_type_counts(statements: list[dict]) -> Counter:
    return Counter(s["type"] for s in statements)
