"""
scripts.py
Writing systems for the script-generalisation test: the trained model (Latin
lowercase only) is evaluated, without any retraining, on secret masks written
in other alphabets.

    latin     a-z (26)                       DejaVu Sans Bold   the training set
    cyrillic  Russian lowercase a-ya + yo (33) DejaVu Sans Bold  same font as
              training, so only the glyph shapes change
    hangul    Korean jamo: 19 consonants + 21 vowels (40)   Noto Sans CJK KR Bold

Every script is rendered by util.word2mask on the same cell grid. Wide glyphs
take two cell slots, decided by measuring them against the widest single-slot
Latin letter (see util.glyph_spans), so every script sits on the grid the way
the training masks do; for Latin the original m/w rule is kept, so its masks
are identical to every earlier run.

Adding a script = one entry in SCRIPTS. A font is looked up in this order:
config.script_fonts[name], then the candidates listed here.

Check fonts and preview masks before any GPU run:

    python scripts.py --check
writes results/script_check/: one preview mask per script and cell height, and
check.md with glyph coverage, wide glyphs, bits per character, expected
characters per mask, and glyph-complexity statistics (ink fraction, stroke
width) that the analysis relates the results to.
"""
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageDraw

from util import _load_font, glyph_spans, word2mask

# Hangul compatibility jamo, the 40 letters of the modern alphabet in the
# usual order. The compatibility block also holds 11 compound final consonants
# (ㄳ ㄵ ㄶ ㄺ ...); they are not letters of the alphabet and are left out.
_HANGUL_CONSONANTS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"            # 19
_HANGUL_VOWELS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"        # 21

# Only the Korean script needs these; the paper's tables use latin_sq, which
# falls back to DejaVuSans-Bold and needs no extra font. Point ECSNET_FONTS at
# a directory of Noto fonts to run the non-Latin scripts.
_FONT_DIR = os.environ.get("ECSNET_FONTS", "")
# A None entry here means "fall back to DejaVu", which would render Hangul as
# empty boxes, so the ECSNET_FONTS candidates are added only when that variable
# is set rather than being inserted as None.
_CJK_KR_FONTS = [
    *([os.path.join(_FONT_DIR, "NotoSansKR-Bold.otf"),
       os.path.join(_FONT_DIR, "NotoSansCJKkr-Bold.otf")] if _FONT_DIR else []),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc#1",      # apt fonts-noto-cjk; face 1 = KR
]


@dataclass
class Script:
    name: str
    label: str
    alphabet: str
    double_width: Union[None, set, str]   # None = Latin m/w rule; "auto" = measured
    fonts: Sequence[Optional[str]]        # candidates; None = config.font_path / DejaVu Sans Bold

    @property
    def bits_per_char(self) -> float:
        return math.log2(len(self.alphabet))


_LATIN = "abcdefghijklmnopqrstuvwxyz"

SCRIPTS = {
    "latin": Script("latin", "Latin (English)", _LATIN,
                    None, [None]),
    # Same 26 letters, same font, but EVERY character takes two cell slots, so
    # the glyph box is square (cell_height x cell_height) instead of half-width.
    # Not a different writing system -- a different grid geometry for the same
    # one -- but it rides the script mechanism because mask_kwargs() is what
    # keeps word2mask and build_templates agreeing on the layout, and that
    # agreement is exactly what must not break.
    #
    # Rationale: at cell 64 a half-width slot is 4 latent columns after the VAE's
    # 8x downsample, but the measured ink widths are a 4.2, o 4.4, b 4.6 columns
    # -- most letters overflow their slot and bleed into their neighbours. 'm'
    # 6.6 and 'w' already took two slots, fit with margin, and lead the accuracy
    # table at 0.991 / 0.935 against a 0.783 mean. Square cells give every
    # letter that margin. Note this removes CROSSTALK, not aliasing: font_size
    # follows cell_height alone, so the glyph itself is unchanged, and the
    # l/i/j confusions (11% of error, a dot at 2.1 columns) are untouched.
    # Capacity cost at cell 64: 120 -> 64 characters. bits_per_char unchanged
    # at log2(26) = 4.70.
    "latin_sq": Script("latin_sq", "Latin (English), square cells", _LATIN,
                       set(_LATIN), [None]),
    "cyrillic": Script("cyrillic", "Cyrillic (Russian)",
                       "абвгдеёжзийклмнопрстуфхцчшщъыьэюя", "auto", [None]),
    "hangul": Script("hangul", "Hangul jamo (Korean)",
                     _HANGUL_CONSONANTS + _HANGUL_VOWELS, "auto", _CJK_KR_FONTS),
}
assert len(SCRIPTS["cyrillic"].alphabet) == 33
assert len(SCRIPTS["hangul"].alphabet) == 40


def resolve_font(script: Script, cfg=None) -> Optional[str]:
    """The font file this script is rendered with, or None for the default.

    Raises if a non-default font is required and none of the candidates exists:
    silently falling back to DejaVu would render empty boxes for Hangul.
    """
    override = getattr(cfg, "script_fonts", {}) or {}
    if override.get(script.name):
        return override[script.name]
    for f in script.fonts:
        if f is None:
            return getattr(cfg, "font_path", None)
        if os.path.isfile(f.partition("#")[0]):
            return f
    raise FileNotFoundError(
        f"no font for script '{script.name}'. Tried: {list(script.fonts)}. "
        f"Install one (e.g. apt-get install -y fonts-noto-cjk) or set "
        f"config.script_fonts['{script.name}'].")


def get_script(name: Optional[str], cfg=None):
    """(Script, font) for a script name; None means the training script."""
    sc = SCRIPTS[name or "latin"]
    return sc, resolve_font(sc, cfg)


_SCALE_CACHE = {}


def font_scale(name: Optional[str], cfg=None) -> float:
    """Font-size factor that normalises the FONT, not the script.

    Fonts place glyphs differently inside the em box, so the same point size
    in Noto Sans CJK and in DejaVu Sans is not the same visual size. The scale
    makes the Latin letters of this script's font as tall (median) as those of
    the training font; the script's own glyphs then come out at whatever size
    that font gives them relative to Latin -- their natural size. Scripts
    sharing the training font (Cyrillic) get 1.0 by construction: matching
    the scripts' own glyph heights instead would inflate Cyrillic, whose
    lowercase has almost no ascenders or descenders.
    """
    sc, font = get_script(name, cfg)
    if font == getattr(cfg, "font_path", None):
        return 1.0
    key = font
    if key not in _SCALE_CACHE:
        size = 200
        probe = ImageDraw.Draw(Image.new("L", (8, 8)))
        latin = SCRIPTS["latin"].alphabet

        def heights(chars, f):
            out = []
            for c in chars:
                b = probe.textbbox((0, 0), c, font=f)
                out.append(b[3] - b[1])
            return out
        ref = np.median(heights(latin, _load_font(None, size)))
        f = _load_font(font, size)
        s = float(ref / np.median(heights(latin, f)))
        tallest = max(heights(sc.alphabet, f)) * s
        s = s * min(1.0, (size / 0.8) / tallest)     # tallest glyph <= cell height
        _SCALE_CACHE[key] = round(s, 3)
    return _SCALE_CACHE[key]


def mask_kwargs(name: Optional[str], cfg=None) -> dict:
    """Keyword arguments for word2mask / build_templates for this script."""
    sc, font = get_script(name, cfg)
    return {"font_path": font,
            "alphabet": None if sc.name == "latin" else sc.alphabet,
            "double_width": sc.double_width,
            "font_scale": font_scale(name, cfg)}


# ------------------------------------------------------------------ checks ---
def _render(ch, font, w, h):
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    b = d.textbbox((0, 0), ch, font=font)
    d.text(((w - (b[2] - b[0])) / 2 - b[0], (h - (b[3] - b[1])) / 2 - b[1]),
           ch, fill=255, font=font)
    return (np.asarray(img) > 127)


def coverage(script: Script, font_path, size=64):
    """Characters the font cannot draw. Uses the font's cmap when fontTools is
    installed; otherwise compares each glyph against the font's .notdef box."""
    path, _, idx = str(font_path or "").partition("#")
    if path:
        try:
            from fontTools.ttLib import TTFont
            f = TTFont(path, fontNumber=int(idx) if idx else 0, lazy=True)
            cmap = f.getBestCmap()
            return [c for c in script.alphabet if ord(c) not in cmap]
        except ImportError:
            pass
    font = _load_font(font_path, size)
    notdef = _render("", font, size * 2, size * 2)
    return [c for c in script.alphabet
            if not _render(c, font, size * 2, size * 2).any()
            or np.array_equal(_render(c, font, size * 2, size * 2), notdef)]


def glyph_stats(script: Script, font_path, cell, scale=1.0):
    """Per-glyph complexity at this cell height, as rendered in its box.

    ink   : fraction of the glyph's box that is white
    width : mean stroke width in pixels, estimated as 2 * area / perimeter
            (exact for a long uniform stroke). The quantity to compare with the
            8-pixel VAE downsampling and the 16-pixel transformer patch.
    """
    font = _load_font(font_path, int(cell * 0.8 * scale))
    cw = cell // 2
    spans = glyph_spans(script.alphabet, font, cw, script.double_width)
    inks, widths, ratio = [], [], []
    for c in script.alphabet:
        g = _render(c, font, cw * spans[c], cell)
        a = g.sum()
        if a == 0:
            continue
        inner = g[1:-1, 1:-1] & g[:-2, 1:-1] & g[2:, 1:-1] & g[1:-1, :-2] & g[1:-1, 2:]
        perim = a - inner.sum()
        inks.append(a / g.size)
        widths.append(2 * a / max(perim, 1))
        b = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), c, font=font)
        ratio.append((b[2] - b[0]) / (cw * spans[c]))
    return {"ink": float(np.mean(inks)), "stroke_px": float(np.mean(widths)),
            "wide": [c for c in script.alphabet if spans[c] == 2],
            "max_fill": float(np.max(ratio))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--scripts", nargs="+", default=list(SCRIPTS))
    ap.add_argument("--cells", type=int, nargs="+", default=[128, 64, 32])
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--n_sim", type=int, default=300)
    ap.add_argument("--out", default="results/script_check")
    a = ap.parse_args()
    try:
        from config import get_config
        cfg = get_config()
    except Exception:
        cfg = None
    os.makedirs(a.out, exist_ok=True)
    L = ["# Script check\n",
         "| script | font | scale | K | bits/char | cell | wide glyphs | chars/mask | bits/mask | "
         "mask white | glyph ink | stroke px | max glyph/slot width |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = True
    for name in a.scripts:
        sc, font = get_script(name, cfg)
        miss = coverage(sc, font)
        if miss:
            ok = False
            print(f"[FAIL] {name}: font {font} cannot draw {''.join(miss)}")
        for cell in a.cells:
            kw = mask_kwargs(name, cfg)
            n_chars, white = [], []
            for s in range(a.n_sim):
                m, letters, _ = word2mask(a.image_size, cell, seed=s, **kw)
                n_chars.append(len(letters))
                white.append((np.asarray(m) > 127).mean())
                if s == 0:
                    m.save(os.path.join(a.out, f"{name}_cell{cell}.png"))
            st = glyph_stats(sc, font, cell, kw["font_scale"])
            nc = float(np.mean(n_chars))
            L.append(f"| {name} | {os.path.basename(str(font)) if font else 'default'} | "
                     f"{kw['font_scale']:g} | {len(sc.alphabet)} | {sc.bits_per_char:.2f} | {cell} | "
                     f"{''.join(st['wide']) or '-'} | {nc:.1f} | {nc * sc.bits_per_char:.0f} | "
                     f"{np.mean(white):.3f} | {st['ink']:.3f} | {st['stroke_px']:.1f} | "
                     f"{st['max_fill']:.2f} |")
    L.append("\nstroke px is to be read against the 8-px VAE downsampling and the "
             "16-px transformer patch. 'max glyph/slot width' above 1 means glyphs "
             "overhang their slot; the training masks (latin) do, by design, and the "
             "other scripts are laid out to overhang no more than they do.")
    text = "\n".join(L)
    print(text)
    with open(os.path.join(a.out, "check.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n{'ALL FONTS OK' if ok else 'MISSING GLYPHS -- fix the font first'}; "
          f"previews in {a.out}/")


if __name__ == "__main__":
    main()
