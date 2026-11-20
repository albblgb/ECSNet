"""
char_metrics.py
把预测出的 mask 按 word2mask 的格子布局切开，每格和 26 个标准字母模板匹配，
得到字符级准确率。

为什么不用 OCR：字符位置、字体、大小、字符集在这里全部已知，OCR 要解决的
不确定性一个都不存在；而且 OCR 漏检一个字符会让后面整串错位，把单个错误
放大成全序列错误。按已知格子切开做模板匹配没有这个问题。

模板匹配用的是 IoU（交并比），因为预测和模板都是二值图，IoU 比像素点积
更能反映形状吻合程度，也不会因为笔画粗细差异被拉偏。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from util import _load_font, glyph_spans


def build_templates(cell_width, cell_height, font_path=None, alphabet=None,
                    double_width=None, font_scale=1.0):
    """
    为每个字母渲染一张标准模板，渲染方式必须和 word2mask 完全一致，
    否则匹配的是"渲染差异"而不是"识别正确与否"。

    alphabet / double_width / font_scale 与 word2mask 同义（见 scripts.py）；
    默认值即训练用的 a-z、m/w 双宽、原字号。

    返回 {char: (template_array, span)}，span 是该字母占几个格子，
    由 util.glyph_spans 决定，和 word2mask 用的是同一个函数。
    """
    if alphabet is None:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
    font_size = int(cell_height * 0.8 * font_scale)
    font = _load_font(font_path, font_size)
    spans = glyph_spans(alphabet, font, cell_width, double_width)

    templates = {}
    for ch in alphabet:
        span = spans[ch]
        box_w = cell_width * span

        img = Image.new("L", (box_w, cell_height), color=0)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), ch, font=font)
        glyph_w = bbox[2] - bbox[0]
        glyph_h = bbox[3] - bbox[1]
        cx = (box_w - glyph_w) / 2 - bbox[0]
        cy = (cell_height - glyph_h) / 2 - bbox[1]
        draw.text((cx, cy), ch, fill=255, font=font)

        templates[ch] = ((np.array(img) > 127).astype(np.float32), span)
    return templates


def _iou(a, b):
    inter = np.logical_and(a > 0.5, b > 0.5).sum()
    union = np.logical_or(a > 0.5, b > 0.5).sum()
    return inter / union if union > 0 else 0.0


def match_cell(patch, templates, span):
    """
    把一个格子的预测 patch 和所有 span 匹配的模板比对，返回 (best_char, best_iou)。

    只和相同 span 的模板比：格子布局是已知的，我们知道这一格是单宽还是双宽，
    没必要让 'm' 去和 'a' 竞争。
    """
    best_char, best_iou = None, -1.0
    for ch, (tmpl, tspan) in templates.items():
        if tspan != span:
            continue
        if tmpl.shape != patch.shape:
            continue
        score = _iou(patch, tmpl)
        if score > best_iou:
            best_iou, best_char = score, ch
    return best_char, best_iou


def char_accuracy(pred_mask, layout, cell_width, cell_height, templates,
                  blank_iou_threshold=0.05):
    """
    pred_mask : [H, W] 二值预测（0/1 或 bool）
    layout    : word2mask 返回的 layout，每项 {row, col_start, col_span, char}
    返回 dict，含准确率、逐格结果、以及匹配置信度分布。

    blank_iou_threshold: 若最佳匹配的 IoU 低于此值，判为"没认出来"（记为空），
    而不是强行给一个最相似的字母。这样能区分"认错了"和"根本没提取到"，
    两者对隐写的含义不同。
    """
    pred = (np.asarray(pred_mask) > 0.5).astype(np.float32)

    n_correct, n_blank = 0, 0
    per_cell, ious = [], []

    for entry in layout:
        r, c0, span, true_char = entry["row"], entry["col_start"], entry["col_span"], entry["char"]
        y0 = r * cell_height
        x0 = c0 * cell_width
        patch = pred[y0:y0 + cell_height, x0:x0 + cell_width * span]

        if patch.shape != (cell_height, cell_width * span):
            continue  # 边界越界，跳过

        pred_char, iou = match_cell(patch, templates, span)
        ious.append(iou)

        if iou < blank_iou_threshold:
            pred_char = None
            n_blank += 1
        elif pred_char == true_char:
            n_correct += 1

        per_cell.append({"true": true_char, "pred": pred_char, "iou": float(iou),
                          "row": r, "col_start": c0, "span": span})

    n_total = len(per_cell)
    return {
        "char_accuracy": n_correct / n_total if n_total else 0.0,
        "blank_rate": n_blank / n_total if n_total else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "n_chars": n_total,
        "n_correct": n_correct,
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Soft matching
# ---------------------------------------------------------------------------
# The hard matcher above throws information away twice. It binarises R's
# output at 0.5, so the confidence that separates a thin 'i' from an 'l' is
# lost before matching starts; and it compares against ideal glyphs at a fixed
# position, while R's output is systematically distorted (strokes thicken,
# dots merge into stems, the whole glyph can sit a pixel or two off).
#
# The soft matcher keeps the probability map, blurs the templates so a slightly
# thicker or displaced stroke still overlaps, allows small shifts, and scores
# with normalised cross-correlation. It also returns a ranked candidate list per
# cell with scores, which a language-model decoder can use later.

def build_soft_templates(cell_width, cell_height, font_path=None,
                         alphabet=None, blur_sigma=None, shift=None,
                         double_width=None, font_scale=1.0):
    """Blurred, shifted templates, flattened and normalised for NCC.

    Returns {span: (chars, bank)} where bank is [n_chars, n_shifts, D], each row
    zero-mean and unit-norm, so a dot product with a normalised patch is the
    normalised cross-correlation.
    """
    from scipy.ndimage import gaussian_filter, shift as nd_shift

    hard = build_templates(cell_width, cell_height, font_path, alphabet,
                           double_width, font_scale)
    sigma = blur_sigma if blur_sigma is not None else max(0.5, cell_height / 32)
    r = shift if shift is not None else max(1, cell_height // 16)
    offsets = [(dy, dx) for dy in (-r, 0, r) for dx in (-r, 0, r)]

    banks = {}
    for ch, (tmpl, span) in hard.items():
        t = gaussian_filter(tmpl, sigma)
        rows = []
        for dy, dx in offsets:
            s = nd_shift(t, (dy, dx), order=0, mode="constant", cval=0.0).ravel()
            s = s - s.mean()
            n = np.linalg.norm(s)
            rows.append(s / n if n > 0 else s)
        banks.setdefault(span, ([], []))
        banks[span][0].append(ch)
        banks[span][1].append(np.stack(rows))
    return {span: (chars, np.stack(rows)) for span, (chars, rows) in banks.items()}


def char_accuracy_soft(prob_map, layout, cell_width, cell_height, soft_templates,
                       top_k=3, blank_mass=0.02, temperature=0.05):
    """Character accuracy from R's probability map, by soft template matching.

    prob_map : [H, W] probabilities in [0, 1] (sigmoid of R's logits), NOT
               binarised.
    blank_mass : a cell whose mean probability is below this is reported as
               blank -- nothing was extracted there -- rather than forced onto
               the least-bad template.
    temperature : turns NCC scores into a distribution over characters for the
               top-k list. Only the ranking matters for accuracy; the
               probabilities are for a downstream language-model decoder.

    Returns the same keys as char_accuracy, plus top-k accuracy and, per cell,
    the ranked candidates with probabilities.
    """
    p = np.asarray(prob_map, dtype=np.float32)
    n_correct, n_topk, n_blank = 0, 0, 0
    per_cell, scores_best = [], []

    # Group cells by span so each group is one matrix product.
    groups = {}
    for idx, e in enumerate(layout):
        groups.setdefault(e["col_span"], []).append(idx)

    results = [None] * len(layout)
    for span, idxs in groups.items():
        if span not in soft_templates:
            continue
        chars, bank = soft_templates[span]              # bank [C, S, D]
        C, S, D = bank.shape
        patches, keep = [], []
        for i in idxs:
            e = layout[i]
            y0, x0 = e["row"] * cell_height, e["col_start"] * cell_width
            patch = p[y0:y0 + cell_height, x0:x0 + cell_width * span]
            if patch.shape != (cell_height, cell_width * span):
                continue
            patches.append(patch.ravel())
            keep.append(i)
        if not patches:
            continue
        P = np.stack(patches)                            # [N, D]
        mass = P.mean(axis=1)
        Pc = P - P.mean(axis=1, keepdims=True)
        Pn = Pc / np.maximum(np.linalg.norm(Pc, axis=1, keepdims=True), 1e-8)
        ncc = (Pn @ bank.reshape(C * S, D).T).reshape(-1, C, S).max(axis=2)  # [N, C]

        for row, i in enumerate(keep):
            s = ncc[row]
            order = np.argsort(-s)
            z = (s - s.max()) / temperature
            prob = np.exp(z) / np.exp(z).sum()
            cands = [(chars[j], float(prob[j]), float(s[j])) for j in order[:top_k]]
            results[i] = (cands, float(mass[row]))

    for i, e in enumerate(layout):
        if results[i] is None:
            continue
        cands, mass = results[i]
        true_char = e["char"]
        if mass < blank_mass:
            pred_char = None
            n_blank += 1
        else:
            pred_char = cands[0][0]
            if pred_char == true_char:
                n_correct += 1
        if any(c == true_char for c, _, _ in cands) and mass >= blank_mass:
            n_topk += 1
        scores_best.append(cands[0][2])
        per_cell.append({"true": true_char, "pred": pred_char,
                         "ncc": cands[0][2], "candidates": cands,
                         "row": e["row"], "col_start": e["col_start"],
                         "span": e["col_span"]})

    n = len(per_cell)
    return {
        "char_accuracy": n_correct / n if n else 0.0,
        "topk_accuracy": n_topk / n if n else 0.0,
        "blank_rate": n_blank / n if n else 0.0,
        "mean_ncc": float(np.mean(scores_best)) if scores_best else 0.0,
        "n_chars": n,
        "n_correct": n_correct,
        "per_cell": per_cell,
    }


def format_strings(result):
    """把逐格结果拼成两行字符串，方便肉眼对照。未识别的格子用 '_' 占位。"""
    true_s = "".join(c["true"] for c in result["per_cell"])
    pred_s = "".join(c["pred"] if c["pred"] else "_" for c in result["per_cell"])
    return true_s, pred_s