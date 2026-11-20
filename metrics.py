"""
metrics.py
Image-quality and message-accuracy metrics for the ECSNet experiments.

PSNR and SSIM are implemented here rather than pulled from skimage so they run
on the GPU in batch and so their conventions are pinned: SSIM in particular has
enough free parameters (window, sigma, data range, per-channel vs mean) that two
papers quoting "SSIM" can differ by 0.02 without either being wrong. The
implementation below is the standard Wang et al. formulation with an 11x11
Gaussian of sigma 1.5, averaged over channels, which is what skimage and
torchmetrics both default to.

LPIPS needs pretrained weights and so is optional; the metric is returned as NaN
with one clear message if the package is absent, rather than silently dropping
the column.
"""
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Image quality.  All take images in [-1, 1], shape [B, 3, H, W].
# ---------------------------------------------------------------------------
def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x.float().clamp(-1, 1) + 1) / 2


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB, [B]. Data range 1.0 after the [-1,1] -> [0,1] map."""
    a, b = _to01(a), _to01(b)
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    # An exact match gives infinite PSNR; clamp so one identical pair does not
    # turn the whole dataset mean into inf.
    return 10 * torch.log10(1.0 / mse.clamp(min=1e-10))


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    c = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5) -> torch.Tensor:
    """Per-image SSIM, [B]. Wang et al., 11x11 Gaussian, averaged over channels."""
    a, b = _to01(a), _to01(b)
    B, C, H, W = a.shape
    w = _gaussian_window(window_size, sigma, a.device, a.dtype)
    w = w.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def filt(x):
        return F.conv2d(x, w, padding=pad, groups=C)

    mu_a, mu_b = filt(a), filt(b)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = filt(a * a) - mu_a2
    sb = filt(b * b) - mu_b2
    sab = filt(a * b) - mu_ab

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return m.mean(dim=(1, 2, 3))


class LPIPS:
    """Lazy LPIPS wrapper.

    Optional on purpose: it needs pretrained VGG/AlexNet weights, which on a
    network that cannot reach the usual hosts is a separate problem from
    running the evaluation. When unavailable the metric reports NaN and says
    how to enable it, rather than being quietly omitted from the table.
    """

    def __init__(self, device, net: str = "alex"):
        self.ok = False
        self.why = ""
        try:
            import lpips as _l
            self.model = _l.LPIPS(net=net).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.ok = True
        except Exception as e:
            self.why = f"{type(e).__name__}: {e}"

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self.ok:
            return torch.full((a.shape[0],), float("nan"), device=a.device)
        # lpips expects [-1, 1], which is what the pipeline already uses.
        return self.model(a.float().clamp(-1, 1), b.float().clamp(-1, 1)).flatten()


class CLIPScore:
    """CLIP score between an image and its caption: 100 * max(cos, 0).

    The torchmetrics / Hessel et al. convention without the 2.5 rescaling, so
    numbers are comparable with most text-to-image and inpainting papers that
    quote "CLIP score". PSNR/SSIM/LPIPS ask whether the result looks like the
    original; inpainting is supposed to produce something plausible rather than
    the original, so this asks the other question -- does the image match its
    text.

    Lazy and optional, like LPIPS: a missing model gives NaN plus one message.
    """

    def __init__(self, device, path: Optional[str]):
        self.ok, self.why, self.device = False, "", device
        if not path:
            self.why = "no clip_model_path configured"
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(path).to(device).eval()
            self.proc = CLIPProcessor.from_pretrained(path)
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.ok = True
        except Exception as e:
            self.why = f"{type(e).__name__}: {e}"

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        if not self.ok:
            return torch.full((images.shape[0],), float("nan"), device=images.device)
        arr = ((images.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1)
        pil = [x.cpu().numpy() for x in arr]
        inp = self.proc(text=list(texts), images=pil, return_tensors="pt",
                        padding=True, truncation=True).to(self.device)
        # The full forward, not get_image_features / get_text_features: those
        # return a tensor in transformers 4.x but an output object in 5.x.
        # image_embeds / text_embeds on the forward output are the projected
        # embeddings in both, which is what CLIP score is defined on.
        out = self.model(input_ids=inp["input_ids"],
                         attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"])
        img, txt = out.image_embeds.float(), out.text_embeds.float()
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return (100 * (img * txt).sum(-1).clamp(min=0)).float()


    # ---- pieces for the editing experiment ----
    @torch.no_grad()
    def embed_images(self, images) -> torch.Tensor:
        """Unit-norm image embeddings. `images` is a list of [3,h,w] tensors in
        [-1,1] (sizes may differ -- crops) or a batch tensor."""
        if not self.ok:
            return None
        arrs = [((x.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0)
                .cpu().numpy() for x in images]
        pv = self.proc(images=arrs, return_tensors="pt")["pixel_values"].to(self.device)
        # vision tower + projection is what get_image_features computes; called
        # directly so the result is a tensor in transformers 4.x and 5.x alike.
        e = self.model.visual_projection(self.model.vision_model(pixel_values=pv).pooler_output)
        return torch.nn.functional.normalize(e.float(), dim=-1)

    @torch.no_grad()
    def embed_texts(self, texts) -> torch.Tensor:
        if not self.ok:
            return None
        t = self.proc(text=list(texts), return_tensors="pt", padding=True,
                      truncation=True).to(self.device)
        e = self.model.text_projection(self.model.text_model(
            input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output)
        return torch.nn.functional.normalize(e.float(), dim=-1)


def crop_to_mask(images: torch.Tensor, masks: torch.Tensor, margin: float = 0.1,
                 min_size: int = 32):
    """Crop each image to its mask's bounding box, enlarged by `margin`.

    The editing metrics ask whether the NEW OBJECT is there. On the full frame,
    CLIP is dominated by the background, which an object swap leaves intact, so
    a failed edit and a successful one score almost the same.
    """
    out = []
    H, W = images.shape[-2:]
    for x, m in zip(images, masks):
        ys, xs = torch.nonzero(m[0] > 0.5, as_tuple=True)
        if ys.numel() == 0:
            out.append(x)
            continue
        y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
        dy = max(int((y1 - y0) * margin), (min_size - (y1 - y0)) // 2, 0)
        dx = max(int((x1 - x0) * margin), (min_size - (x1 - x0)) // 2, 0)
        y0, y1 = max(0, y0 - dy), min(H, y1 + dy + 1)
        x0, x1 = max(0, x0 - dx), min(W, x1 + dx + 1)
        out.append(x[:, y0:y1, x0:x1])
    return out


def dilate(masks: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return masks
    return F.max_pool2d(masks.float(), kernel_size=2 * px + 1, stride=1, padding=px)


def background_metrics(a: torch.Tensor, b: torch.Tensor, masks: torch.Tensor,
                       lpips_fn=None, dilate_px: int = 8):
    """PSNR / SSIM / LPIPS over the region OUTSIDE the edit mask.

    The mask is dilated by one latent cell (8 px): inpainting in latent space
    legitimately changes a thin band around the mask, and counting it would
    charge the model for the latent resolution rather than for touching the
    background.

    PSNR is computed on the kept pixels only. SSIM and LPIPS need whole images,
    so for them the masked region of `a` is replaced by `b` before comparing --
    the excluded area then contributes no difference.
    """
    keep = 1.0 - dilate(masks, dilate_px)
    a01, b01 = _to01(a), _to01(b)
    se = ((a01 - b01) ** 2 * keep).sum(dim=(1, 2, 3))
    n = (keep.sum(dim=(1, 2, 3)) * a.shape[1]).clamp(min=1)
    bg_psnr = 10 * torch.log10(1.0 / (se / n).clamp(min=1e-10))
    comp = a.float() * keep + b.float() * (1 - keep)
    bg_ssim = ssim(comp, b)
    bg_lpips = lpips_fn(comp, b) if lpips_fn is not None else \
        torch.full_like(bg_psnr, float("nan"))
    return bg_psnr, bg_ssim, bg_lpips


# ---------------------------------------------------------------------------
# Mask accuracy
# ---------------------------------------------------------------------------
def mask_f1_iou(pred_logits: torch.Tensor, target: torch.Tensor):
    """Per-image (F1, IoU) on the white pixels, [B] each.

    Scored on the white (letter) pixels only. Masks are mostly black, so plain
    pixel accuracy sits above 0.75 for a model that outputs nothing at all.
    """
    pred = (pred_logits > 0).float()
    t = target.float()
    dims = (1, 2, 3)
    tp = (pred * t).sum(dims)
    fp = (pred * (1 - t)).sum(dims)
    fn = ((1 - pred) * t).sum(dims)
    p = tp / (tp + fp).clamp(min=1)
    r = tp / (tp + fn).clamp(min=1)
    f1 = 2 * p * r / (p + r).clamp(min=1e-8)
    iou = tp / (tp + fp + fn).clamp(min=1)
    return f1, iou


def white_fraction(pred_logits: torch.Tensor) -> torch.Tensor:
    """Fraction of pixels predicted white, [B].

    This is what separates a genuine rejection from a model that is merely
    wrong: a rejection drives it to ~0, while a wrong answer keeps emitting
    plenty of white. F1 alone cannot tell the two apart.
    """
    return (pred_logits > 0).float().mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Message accuracy
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """Edit distance, iterative with a single row of state."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,        # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, edit distance normalised by reference length."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def summarise(values: List[float]) -> Dict[str, float]:
    """mean / std / median / min / max, NaN-safe.

    NaN-safe matters: a dataset where LPIPS is unavailable, or where some
    samples carry no characters, should still produce a table rather than a
    column of NaN.
    """
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "min", "max")} | {"n": 0}
    return {"mean": float(v.mean()), "std": float(v.std()),
            "median": float(np.median(v)), "min": float(v.min()),
            "max": float(v.max()), "n": int(v.size)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(4, 3, 64, 64) * 2 - 1

    print("Identity:")
    print(f"  PSNR {psnr(x, x).tolist()}  (clamped, not inf)")
    print(f"  SSIM {[round(v, 6) for v in ssim(x, x).tolist()]}  (expect 1.0)")

    noisy = (x + 0.1 * torch.randn_like(x)).clamp(-1, 1)
    print(f"\nAfter sigma=0.1 noise:  PSNR {psnr(x, noisy).mean():.2f} dB  "
          f"SSIM {ssim(x, noisy).mean():.4f}")

    # PSNR must match the textbook formula computed independently.
    a01, b01 = _to01(x), _to01(noisy)
    ref = 10 * math.log10(1.0 / ((a01[0] - b01[0]) ** 2).mean().item())
    print(f"  PSNR[0] {psnr(x, noisy)[0]:.6f} vs closed form {ref:.6f}")

    # SSIM of an image against a blurred copy must fall between 0 and 1 and
    # drop monotonically with blur.
    import torch.nn.functional as _F
    for k in (3, 7, 15):
        w = torch.ones(3, 1, k, k) / (k * k)
        blur = _F.conv2d(x, w, padding=k // 2, groups=3)
        print(f"  blur {k:>2}: SSIM {ssim(x, blur).mean():.4f}  "
              f"PSNR {psnr(x, blur).mean():.2f}")

    print("\nMask metrics:")
    t = (torch.rand(4, 1, 32, 32) > 0.8).float()
    perfect = (t * 2 - 1) * 10
    f1, iou = mask_f1_iou(perfect, t)
    print(f"  perfect prediction: F1 {f1.mean():.4f} IoU {iou.mean():.4f} "
          f"white {white_fraction(perfect).mean():.4f}")
    silent = torch.full_like(t, -10.0)
    f1, iou = mask_f1_iou(silent, t)
    print(f"  silent  prediction: F1 {f1.mean():.4f} IoU {iou.mean():.4f} "
          f"white {white_fraction(silent).mean():.4f}")

    print("\nCER:")
    for a, b in [("hello", "hello"), ("hello", "hallo"), ("hello", "h_llo"),
                 ("hello", ""), ("abc", "abcd")]:
        print(f"  {a!r:>9} vs {b!r:<9} -> {cer(a, b):.4f}")

    print("\nsummarise with NaNs:", summarise([1.0, 2.0, float("nan"), 3.0]))
    print("summarise all NaN   :", summarise([float("nan")]))

    l = LPIPS("cpu")
    print(f"\nLPIPS available: {l.ok}" + ("" if l.ok else f"  ({l.why})"))