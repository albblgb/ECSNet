"""
messages.py
Meaningful secret messages instead of random letters.

The trained model only knows lowercase a-z in DejaVu Sans Bold on a fixed grid,
so a message is normalised to that alphabet -- lowercase, every other
character (punctuation, digits, accents) becomes a word break -- and laid out
on the same grid word by word:

    * letters exactly as word2mask draws them (same font, size, centring,
      m/w take two slots);
    * by default WITHOUT spaces: the letters run on and fill the grid, as in
      training. Empty slots between words were tried and fail: the extractor
      fills them with letters (see text2mask), so word boundaries are left to
      the receiver's decoder (llm_correct.py).

Messages come from a corpus of sentences, several sentences per mask, picked
by the sample's seed, so a run is reproducible and every model sees the same
messages:

    "coco"       COCO train2017 captions (default; on disk already). Captions
                 of other images than the cover, so the message says nothing
                 about the picture it hides in.
    <file.txt>   one sentence per line, e.g. secret-message-style sentences.

text2mask() returns the same (mask, letters, layout) triple as word2mask plus
the message itself, so all downstream metrics work unchanged; word boundaries
are recoverable from the layout (a gap in the slots or a new row).
"""
import json
import os
import random
import re
from typing import List, Optional

from PIL import Image, ImageDraw

from util import _load_font, glyph_spans

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def normalise(s: str) -> str:
    """Lowercase a-z and single spaces only."""
    s = re.sub(r"['\u2019]", "", s.lower())      # let's -> lets, not "let s"
    s = re.sub(r"[^a-z]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_corpus(source: str, cfg=None, min_words: int = 3) -> List[str]:
    if source == "coco":
        spec = cfg.train_dataset
        path = os.path.join(spec.root, spec.caption_file)
        with open(path) as f:
            caps = [a["caption"] for a in json.load(f)["annotations"]]
    else:
        with open(source, encoding="utf-8") as f:
            caps = [ln for ln in f]
    out, seen = [], set()
    for c in caps:
        n = normalise(c)
        if len(n.split()) >= min_words and n not in seen:
            seen.add(n)
            out.append(n)
    if not out:
        raise ValueError(f"no usable sentences in {source}")
    return out


def text2mask(image_height: int, word_cell_height: int, words: List[str],
              font_path: Optional[str] = None, spaces: bool = False,
              double_width=None):
    """Lay `words` out on the grid.

    spaces=False (default): the letters run on without spaces, row after row,
        exactly like the training masks -- every slot holds a letter, words
        may break across rows, and the grid is filled to the end (the last
        word may be cut). The receiver gets a string of letters and has to
        find the word boundaries itself.
    spaces=True: one empty slot between words, line breaks between words, cut
        at the last whole word. Kept to document why it is not used: the
        extractor never saw an empty slot inside a row in training and draws
        a letter into it (measured: blank slots come out as inked as letter
        slots), so the boundaries do not survive the channel.

    Returns mask (PIL 'L'), letters (list), layout (list of dicts, letters
    only), message (what was placed, words separated by single spaces).
    """
    cell_h, cell_w = word_cell_height, word_cell_height // 2
    n_rows, n_cols = image_height // cell_h, image_height // cell_w
    mask = Image.new("L", (image_height, image_height), 0)
    draw = ImageDraw.Draw(mask)
    font = _load_font(font_path, int(cell_h * 0.8))
    # double_width=None is the original m/w rule; a set (e.g. every letter,
    # for square cells) must match what the model was trained on and what
    # build_templates reads with, or every character scores as an error.
    spans = glyph_spans(ALPHABET, font, cell_w, double_width)

    letters, layout, placed = [], [], []
    row, col = 0, 0

    def put(ch, r, c):
        span = spans[ch]
        b = draw.textbbox((0, 0), ch, font=font)
        cx = c * cell_w + (cell_w * span - (b[2] - b[0])) / 2 - b[0]
        cy = r * cell_h + (cell_h - (b[3] - b[1])) / 2 - b[1]
        draw.text((cx, cy), ch, fill=255, font=font)
        letters.append(ch)
        layout.append({"row": r, "col_start": c, "col_span": span, "char": ch})

    if not spaces:
        placed_n = 0
        stream = "".join(words)
        for ch in stream:
            if col + spans[ch] > n_cols:            # m/w with one slot left:
                row, col = row + 1, 0               # it goes to the next row
            if row >= n_rows:
                break
            put(ch, row, col)
            col += spans[ch]
            placed_n += 1
        out, left = [], placed_n                    # the placed text, with its spaces
        for w in words:
            if left <= 0:
                break
            out.append(w[:left])
            left -= len(w)
        return mask, letters, layout, " ".join(out)

    for w in words:
        width = sum(spans[ch] for ch in w)
        gap = 0 if col == 0 else 1
        if col + gap + width > n_cols:              # next row
            row, col, gap = row + 1, 0, 0
        if row >= n_rows:
            break
        if width > n_cols:                          # longer than a row: split
            remaining = sum(n_cols for _ in range(n_rows - row))
            if width > remaining:
                break
            for ch in w:
                if col + spans[ch] > n_cols:
                    row, col = row + 1, 0
                put(ch, row, col)
                col += spans[ch]
        else:
            col += gap
            for ch in w:
                put(ch, row, col)
                col += spans[ch]
        placed.append(w)
    return mask, letters, layout, " ".join(placed)


class MessageSampler:
    """Deterministic message per seed: random sentences, concatenated."""

    def __init__(self, source: str = "coco", cfg=None, spaces: bool = False):
        self.source, self.spaces = source, spaces
        self.sentences = load_corpus(source, cfg)

    def render(self, image_height, cell, seed, font_path=None, double_width=None):
        rng = random.Random(seed * 1000003 + 17)
        words: List[str] = []
        # Enough text to overfill the grid; text2mask cuts at a whole word.
        need = (image_height // cell) * (image_height // (cell // 2))
        while sum(len(w) + 1 for w in words) < need * 1.5:
            words += self.sentences[rng.randrange(len(self.sentences))].split()
        return text2mask(image_height, cell, words, font_path, self.spaces,
                         double_width=double_width)


def words_from_layout(layout):
    """Word boundaries as the receiver sees them on the grid: a new word starts
    on a new row or after an empty slot. Returns a list of lists of layout
    indices."""
    words, cur = [], []
    for i, e in enumerate(layout):
        if cur:
            p = layout[cur[-1]]
            if e["row"] != p["row"] or e["col_start"] != p["col_start"] + p["col_span"]:
                words.append(cur)
                cur = []
        cur.append(i)
    if cur:
        words.append(cur)
    return words


def blank_slots(layout, image_height, cell):
    """(row, col) of the empty slots between words -- where the receiver must
    read nothing. Trailing slots after the last word of a row are included,
    and rows after the last one are not (the message ended there)."""
    n_cols = image_height // (cell // 2)
    used = {}
    last_row = max(e["row"] for e in layout) if layout else -1
    for e in layout:
        for k in range(e["col_span"]):
            used[(e["row"], e["col_start"] + k)] = True
    return [(r, c) for r in range(last_row + 1) for c in range(n_cols) if (r, c) not in used]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="preview a few message masks")
    ap.add_argument("--source", default="coco")
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="results/message_preview")
    a = ap.parse_args()
    cfg = None
    if a.source == "coco":
        from config import get_config
        cfg = get_config()
    ms = MessageSampler(a.source, cfg)
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(ms.sentences)} sentences")
    for s in range(a.n):
        m, letters, lay, msg = ms.render(512, a.cell, s)
        m.save(os.path.join(a.out, f"msg_cell{a.cell}_{s}.png"))
        print(f"[{s}] {len(letters)} letters, {len(msg.split())} words: {msg}")