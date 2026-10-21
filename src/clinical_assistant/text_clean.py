"""PDF text cleanup: the guideline PDF's symbol font decodes wrongly, and docling drops spaces
in some list items. See notebooks/README.md §1 step 2 for why each fix exists.
"""

import re

from . import config

GLYPH_MAP = {
    "‡": "≥",
    "£": "≤",
    "¼": "=",
    "þ": "+",
    "ª": "©",
    "/C15": "•",
    "\x0f": "•",  # same bullet glyph, decoded as a control character
    "$": "≥",  # in tables; all 47 occurrences are "≥" (e.g. "eGFR $60", "ACR $30 mg/g")
    "–": "-",  # en dash: docling and pypdf disagree, normalise both
    "ﬁ": "fi",
    "ﬂ": "fl",
    "­": "",  # soft hyphen
}


def map_glyphs(text: str) -> str:
    for bad, good in GLYPH_MAP.items():
        text = text.replace(bad, good)
    # re-join words hyphenated across line breaks: "recom- mend" -> "recommend"
    text = re.sub(r"(\w)-[ \n](?=[a-z])", r"\1", text)
    return re.sub(r"[ \t]+", " ", text)  # keep newlines (tables, list items)


class _PypdfReference:
    """Independent pypdf extraction of the whole PDF, used to restore spacing docling drops.

    Built and cached lazily: importing this module (or even parsing statements from the docling
    cache) should not require pypdf to re-read the whole PDF if nothing needs respacing yet.
    """

    def __init__(self):
        self._text = None
        self._positions = None
        self._despaced = None

    def _ensure_loaded(self):
        if self._text is not None:
            return
        if not config.PYPDF_CACHE.exists():
            from pypdf import PdfReader

            config.PYPDF_CACHE.parent.mkdir(parents=True, exist_ok=True)
            config.PYPDF_CACHE.write_text(
                "\n".join((p.extract_text() or "") for p in PdfReader(str(config.GUIDELINE_PATH)).pages),
                encoding="utf-8",
            )
        self._text = map_glyphs(config.PYPDF_CACHE.read_text(encoding="utf-8"))
        self._positions = [i for i, c in enumerate(self._text) if not c.isspace()]
        self._despaced = "".join(self._text[i] for i in self._positions)

    def respace(self, text: str) -> str:
        self._ensure_loaded()

        def fix(match):
            token = match.group(0)
            pos = self._despaced.find(token)
            if pos < 0:
                return token
            spaced = self._text[self._positions[pos] : self._positions[pos + len(token) - 1] + 1]
            return re.sub(r"\s+", " ", spaced)

        # 12+ non-space chars containing a lowercase letter run glued to another word
        return re.sub(r"\S{12,}", fix, text)

    @property
    def full_text(self) -> str:
        self._ensure_loaded()
        return self._text


pypdf_reference = _PypdfReference()


def clean_text(text: str) -> str:
    text = pypdf_reference.respace(map_glyphs(text))
    text = re.sub(r"1\.73 m\s*2\b", "1.73 m2", text)  # "1.73 m 2" -> "1.73 m2"
    return text.strip()
