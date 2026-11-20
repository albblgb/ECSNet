"""
util.py
Utilities for the KFSD project.
Run inside the `sdxl` conda env (needs Pillow: pip install pillow).
"""
import random
import string

from PIL import Image, ImageDraw, ImageFont


def glyph_spans(alphabet, font, cell_width, double_width=None):
    """How many cell slots each character occupies: {char: 1 or 2}.

    double_width : None      -> the original rule, 'm' and 'w' take two slots;
                   a set     -> exactly those characters take two slots;
                   "auto"    -> measured, by the rule the Latin set follows: a
                                glyph takes two slots if it is wider, relative
                                to the font size, than the widest single-slot
                                Latin letter in the training font. Latin glyphs
                                already overhang their half-width slot a little;
                                this keeps other scripts at the same overhang,
                                so they are not laid out more sparsely (or more
                                densely) than the training masks.
    word2mask and the character-metric templates both call this, so the mask
    and the reader always agree on the layout.
    """
    if double_width is None:
        double_width = {"m", "w"}
    if double_width != "auto":
        return {c: (2 if c in double_width else 1) for c in alphabet}
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    limit = _latin_single_slot_ratio() * font.size * 1.03  # 3%: pixel rounding
    spans = {}
    for c in alphabet:
        b = probe.textbbox((0, 0), c, font=font)
        spans[c] = 2 if (b[2] - b[0]) > limit else 1
    return spans


_REF_RATIO = None


def _latin_single_slot_ratio():
    """Width of the widest single-slot Latin letter (a-z without m, w) in the
    training font, as a fraction of the font size."""
    global _REF_RATIO
    if _REF_RATIO is None:
        size = 200
        f = _load_font(None, size)
        probe = ImageDraw.Draw(Image.new("L", (8, 8)))
        w = max(probe.textbbox((0, 0), c, font=f)[2] - probe.textbbox((0, 0), c, font=f)[0]
                for c in string.ascii_lowercase if c not in "mw")
        _REF_RATIO = w / size
    return _REF_RATIO


def word2mask(image_height: int, word_cell_height: int, font_path: str = None,
              seed: int = None, alphabet=None, double_width=None,
              font_scale: float = 1.0):
    """
    Generate a black-background mask image with a randomly sampled English
    letter typed into each "word cell" in white.

    The image is square: image_width == image_height.
    Each cell normally holds one lowercase English character (a-z), with
    cell_width = word_cell_height / 2 (cell_height = 2 * cell_width, per spec).
    Exception: 'm' and 'w' are visually wider, so they occupy TWO adjacent
    cell slots (box width = cell_height, i.e. box is square) instead of one.
    Because a double-width character needs a free slot to its right, 'm'/'w'
    are never sampled when only one slot remains at the end of a row.

    Example: image_height=512, word_cell_height=64
        -> cell_width = 32
        -> n_rows = 512 / 64 = 8
        -> n_cols = 512 / 32 = 16  (slots per row, before accounting for
           double-width characters -- a row containing any 'm'/'w' will end
           up with fewer than 16 characters, since each 'm'/'w' consumes 2 slots)

    Parameters
    ----------
    image_height : int
        Height (and width) of the output mask image, in pixels.
    word_cell_height : int
        Height of each individual character cell, in pixels.
        Must evenly divide image_height, and word_cell_height // 2 must
        evenly divide image_height as well (since image is square).
    font_path : str, optional
        Path to a .ttf/.otf font file. If None, tries a common system font
        (DejaVuSans-Bold), falling back to PIL's built-in bitmap font
        (not recommended -- it does not scale, so provide a real TTF for
        anything beyond quick testing).
    seed : int, optional
        Random seed, for reproducible mask generation.
    alphabet : str | list[str], optional
        Characters to sample from. None = lowercase a-z (the training set).
    double_width : None | set | "auto"
        Which characters take two slots; see glyph_spans. None keeps the
        original 'm'/'w' rule, so the default call reproduces every earlier mask
        exactly.
    font_scale : float
        Multiplies the font size (0.8 * cell height). Fonts draw scripts at
        different sizes within the em box -- CJK fonts draw Hangul jamo much
        smaller than DejaVu draws Latin -- so other scripts are scaled to the
        same glyph height as the training letters (see scripts.py).

    Returns
    -------
    mask : PIL.Image.Image, mode 'L'
        Black background (0), white (255) letters, one per cell.
    letters : list[str]
        The sampled lowercase letters in placement order (row by row,
        left to right within a row). This is the ground-truth secret
        content encoded into the mask, keep it around if you need to
        check decoder accuracy later.
    layout : list[dict]
        One entry per placed character, with keys:
        'row' (int), 'col_start' (int, in cell-width units),
        'col_span' (int, 1 for normal letters, 2 for 'm'/'w'), 'char' (str).
        Needed for decoding, since rows no longer have uniform cell counts
        once 'm'/'w' characters (which consume 2 slots) are involved.
    """
    if seed is not None:
        random.seed(seed)

    image_width = image_height  # square image, per spec

    cell_height = word_cell_height
    cell_width = word_cell_height // 2

    if cell_width <= 0:
        raise ValueError("word_cell_height must be >= 2 (so cell_width >= 1)")
    if image_height % cell_height != 0:
        raise ValueError(
            f"image_height ({image_height}) must be divisible by "
            f"word_cell_height ({word_cell_height})"
        )
    if image_width % cell_width != 0:
        raise ValueError(
            f"image_width ({image_width}) must be divisible by "
            f"cell_width ({cell_width})"
        )

    n_rows = image_height // cell_height
    n_cols = image_width // cell_width

    mask = Image.new("L", (image_width, image_height), color=0)
    draw = ImageDraw.Draw(mask)

    font_size = int(cell_height * 0.8 * font_scale)
    font = _load_font(font_path, font_size)

    if alphabet is None:
        alphabet = string.ascii_lowercase  # a-z only, as in training
    spans = glyph_spans(alphabet, font, cell_width, double_width)
    narrow_alphabet = [c for c in alphabet if spans[c] == 1]

    letters = []
    layout = []
    for row in range(n_rows):
        col = 0
        y0 = row * cell_height
        while col < n_cols:
            remaining = n_cols - col
            if remaining >= 2:
                symbol = random.choice(alphabet)
            else:
                # only one slot left in this row: a double-width glyph would
                # overflow it
                if not narrow_alphabet:
                    break
                symbol = random.choice(narrow_alphabet)

            span = spans[symbol]
            box_width = cell_width * span

            x0 = col * cell_width

            # Center the glyph within its (possibly double-width) box
            bbox = draw.textbbox((0, 0), symbol, font=font)
            glyph_w = bbox[2] - bbox[0]
            glyph_h = bbox[3] - bbox[1]

            cx = x0 + (box_width - glyph_w) / 2 - bbox[0]
            cy = y0 + (cell_height - glyph_h) / 2 - bbox[1]

            draw.text((cx, cy), symbol, fill=255, font=font)

            letters.append(symbol)
            layout.append({"row": row, "col_start": col, "col_span": span, "char": symbol})

            col += span

    return mask, letters, layout


def _load_font(font_path, font_size):
    """Try a user-supplied font, then a common system font, then PIL default.

    A font collection (.ttc) takes its face index after '#', e.g.
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc#1".
    """
    if font_path is not None:
        path, _, idx = str(font_path).partition("#")
        return ImageFont.truetype(path, font_size, index=int(idx) if idx else 0)
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    # last resort: PIL's built-in bitmap font (does not scale with font_size)
    return ImageFont.load_default()


if __name__ == "__main__":
    mask, letters, layout = word2mask(image_height=512, word_cell_height=64, seed=0)
    print(f"Generated mask with {len(letters)} letters (including any 'm'/'w').")
    print("Letters (placement order):", "".join(letters))
    n_double = sum(1 for entry in layout if entry["col_span"] == 2)
    print(f"Double-width ('m'/'w') characters placed: {n_double}")
    mask.save("word_mask_demo.png")
    print("Saved preview to word_mask_demo.png")