import re
from dataclasses import dataclass

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


_DEV_RE = re.compile(r"[\u0900-\u097F]")
_GUJ_RE = re.compile(r"[\u0A80-\u0AFF]")


@dataclass
class Transliterator:
    """Lightweight text transliteration helper for Indic scripts.

    Purpose:
      - Convert Devanagari / Gujarati transcripts to roman script once
      - Keep preprocessing cached so repeated runs don't redo work
    """

    def detect_script(self, text: str) -> str | None:
        if _DEV_RE.search(text):
            return sanscript.DEVANAGARI
        if _GUJ_RE.search(text):
            return sanscript.GUJARATI
        return None

    def transliterate_to_roman(self, text: str) -> str:
        src = self.detect_script(text)
        if src is None:
            return text

        # Use Velthuis (ASCII-friendly) to avoid Unicode diacritics.
        out = transliterate(text, src, sanscript.VELTHUIS)

        # Light normalization to make text more TTS-tokenizer-friendly.
        out = out.lower()
        out = out.replace(".m", "n")
        out = out.replace(".n", "n")
        out = out.replace(".h", "h")
        out = out.replace("~n", "ny")
        out = out.replace("..", ".")
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def transliterate_batch(self, texts: list[str]) -> list[str]:
        return [self.transliterate_to_roman(t) for t in texts]
