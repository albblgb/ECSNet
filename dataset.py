"""
dataset.py
Data for ECSNet: cover image, caption, and a mask that is either a normal
editing mask or a text-based secret mask.

One sample carries exactly one mask, and `is_secret` says which kind it is.
That is what lets train.py send the sample to L_edit or to L_steg without ever
running both branches on the same image: the two mask distributions are
different enough that mixing them inside one loss hides whichever is doing
worse.

The secret mask has two jobs at once -- it is the spatial mask handed to the
transformer, and it is the ground truth the extraction decoder must reproduce.
Both come out of the same tensor, so they cannot drift apart.

Adding Open Images later means writing one more Dataset class and one line in
build_dataset(). Nothing above this file changes: train.py and evaluate.py take
a DatasetSpec and ask for a loader.

Determinism: evaluation must produce the same masks on every run, or the
numbers in the paper are not reproducible and cover/stego pairs saved on one
run will not match those from another. Every per-sample random choice is
therefore drawn from an RNG seeded by (base_seed, index), never from the global
RNG. util.word2mask does call random.seed() globally, which is why none of the
choices here go through the `random` module's global state.
"""
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from util import word2mask


# ---------------------------------------------------------------------------
# Mask generators for the editing branch
# ---------------------------------------------------------------------------
def _box_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """A random rectangle covering 10-40% of the image."""
    frac = rng.uniform(0.10, 0.40)
    ar = rng.uniform(0.5, 2.0)
    area = frac * size * size
    w = int(min(size - 2, max(8, (area * ar) ** 0.5)))
    h = int(min(size - 2, max(8, area / max(w, 1))))
    x = int(rng.integers(0, size - w)) if size > w else 0
    y = int(rng.integers(0, size - h)) if size > h else 0
    m = np.zeros((size, size), np.float32)
    m[y:y + h, x:x + w] = 1.0
    return m


def _brush_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """Free-form brush strokes, the shape an interactive inpainting user draws."""
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(1, 5))):
        x, y = rng.integers(0, size, 2)
        width = int(rng.integers(size // 24, size // 8))
        for _ in range(int(rng.integers(4, 14))):
            nx = int(np.clip(x + rng.integers(-size // 5, size // 5), 0, size - 1))
            ny = int(np.clip(y + rng.integers(-size // 5, size // 5), 0, size - 1))
            d.line([int(x), int(y), nx, ny], fill=255, width=width)
            d.ellipse([nx - width // 2, ny - width // 2,
                       nx + width // 2, ny + width // 2], fill=255)
            x, y = nx, ny
    return (np.array(img) > 127).astype(np.float32)


def _usable_anns(coco, img_id, orig_wh):
    """Instances whose area is 2%-60% of the image: the band every instance
    mask is drawn from, here and in the Open Images loader."""
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    if not ann_ids:
        return []
    ow, oh = orig_wh
    total = max(ow * oh, 1)
    return [a for a in coco.loadAnns(ann_ids)
            if 0.02 * total <= a.get("area", 0) <= 0.60 * total]


def _instance_mask(coco, img_id, size: int, orig_wh, rng: np.random.Generator):
    """One COCO instance polygon, rasterised and resized to `size`.

    Returns (mask, class_name), or None. The class name is what the editing
    experiment replaces.

    Returns None when the image has no annotation worth using -- a mask that is
    nearly empty or nearly the whole frame teaches the editing branch nothing,
    so the caller falls back to a synthetic shape rather than training on it.
    """
    usable = _usable_anns(coco, img_id, orig_wh)
    if not usable:
        return None
    # Pick one at random so the same image does not always give the same mask.
    a = usable[int(rng.integers(0, len(usable)))]
    try:
        m = coco.annToMask(a)
    except Exception:
        return None
    if m.sum() == 0:
        return None
    m = np.array(Image.fromarray((m * 255).astype(np.uint8))
                 .resize((size, size), Image.NEAREST))
    m = (m > 127).astype(np.float32)
    frac = m.mean()
    if frac < 0.01 or frac > 0.70:
        return None
    return m, coco.loadCats(a["category_id"])[0]["name"]


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------
class CocoECSNetDataset(Dataset):
    """COCO2017 as (cover, caption, mask, secret ground truth).

    Parameters
    ----------
    spec : DatasetSpec
    cfg : Config
    force_mask : None | "secret" | "normal"
        None follows cfg.p_secret_mask, which is what training wants.
        Evaluation pins it: experiment 1 needs every sample to be a normal
        inpaint, experiment 2 needs every sample to carry a secret.
    secret_cell_height : int | None
        None samples from cfg.secret_cell_heights. Pinning it is how the
        capacity ablation sweeps density.
    base_seed : int | None
        None makes masks vary per epoch (training). An int makes sample i
        reproducible forever (evaluation).
    limit : int | None
        Keep only the first `limit` images, after a deterministic shuffle.
    """

    def __init__(self, spec, cfg, force_mask: Optional[str] = None,
                 secret_cell_height: Optional[int] = None,
                 base_seed: Optional[int] = None, limit: Optional[int] = None,
                 normal_kind: Optional[str] = None, script: Optional[str] = None,
                 messages=None):
        self.spec = spec
        # Meaningful messages instead of random letters (messages.MessageSampler);
        # None = random letters, as in training. Evaluation only.
        self.messages = messages
        # Writing system / grid geometry of the secret masks. An explicit value
        # wins; otherwise cfg.secret_script, which is what TRAINING uses, so the
        # training masks and the default evaluation masks are the same scheme.
        # Passing a different one is the script-generalisation test.
        self.script = script if script is not None else getattr(
            cfg, "secret_script", None)
        self.cfg = cfg
        self.size = cfg.image_size
        self.force_mask = force_mask
        self.secret_cell_height = secret_cell_height
        self.base_seed = base_seed
        # Pin the normal-mask type ("instance" | "box" | "brush") so each type
        # can be evaluated on its own; None keeps the training mixture.
        self.normal_kind = normal_kind

        self.img_dir = os.path.join(spec.root, spec.images_subdir, spec.split) \
            if spec.images_subdir else os.path.join(spec.root, spec.split)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"image directory not found: {self.img_dir}")

        # ---- captions ----
        cap_path = os.path.join(spec.root, spec.caption_file)
        if not os.path.isfile(cap_path):
            raise FileNotFoundError(f"caption file not found: {cap_path}")
        with open(cap_path) as f:
            cap_json = json.load(f)
        # First caption per image, by annotation id, so the choice is stable
        # across runs and machines.
        caps: Dict[int, str] = {}
        for a in sorted(cap_json["annotations"], key=lambda x: x["id"]):
            caps.setdefault(a["image_id"], a["caption"].strip())
        self.captions = caps
        self.file_names = {im["id"]: im["file_name"] for im in cap_json["images"]}
        self.sizes = {im["id"]: (im["width"], im["height"]) for im in cap_json["images"]}

        # ---- instance annotations (lazy: the train split file is ~450MB) ----
        self._coco = None
        self._ann_path = os.path.join(spec.root, spec.ann_file) if spec.ann_file else None

        ids = sorted(i for i in caps if i in self.file_names)
        random.Random(12345).shuffle(ids)   # fixed, independent of cfg.seed
        if normal_kind == "instance":
            # Keep only images that HAVE a usable instance. Otherwise the
            # loader would quietly fall back to a brush stroke and the
            # "instance" row would be a mixture. Filtering before the limit
            # keeps the order fixed, so this is still "the first n" of a
            # deterministic list.
            if self.coco is None:
                raise FileNotFoundError("normal_kind='instance' needs ann_file")
            ids = [i for i in ids if _usable_anns(self.coco, i, self.sizes[i])]
        if limit is not None:
            ids = ids[:limit]
        self.ids: List[int] = ids

        cum, self._mask_kinds, self._mask_cum = 0.0, [], []
        for k, p in cfg.normal_mask_probs.items():
            cum += p
            self._mask_kinds.append(k)
            self._mask_cum.append(cum)
        self._total_p = cum

    # pycocotools is loaded in the worker that first needs it, not in the parent:
    # the instances json is large and forking it into every worker costs more
    # memory than reloading it.
    @property
    def coco(self):
        if self._coco is None and self._ann_path and os.path.isfile(self._ann_path):
            from pycocotools.coco import COCO
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco = COCO(self._ann_path)
        return self._coco

    def __len__(self):
        return len(self.ids)

    def _instance(self, img_id, orig_wh, rng):
        """One instance mask for this image, or None. Subclasses change only
        where instance masks come from; everything else is shared, so the two
        datasets are evaluated under one protocol."""
        if self.coco is None:
            return None
        return _instance_mask(self.coco, img_id, self.size, orig_wh, rng)

    def _rng(self, i: int):
        if self.base_seed is None:
            # Training: fresh entropy every epoch, so an image is not welded to
            # one mask for the whole run.
            seed = int.from_bytes(os.urandom(4), "little")
        else:
            seed = (self.base_seed * 1000003 + i) % (2 ** 31 - 1)
        return np.random.default_rng(seed), seed

    def _pick_normal_kind(self, rng) -> str:
        u = float(rng.random()) * self._total_p
        for k, c in zip(self._mask_kinds, self._mask_cum):
            if u <= c:
                return k
        return self._mask_kinds[-1]

    def __getitem__(self, i: int):
        img_id = self.ids[i]
        rng, seed = self._rng(i)

        path = os.path.join(self.img_dir, self.file_names[img_id])
        img = Image.open(path).convert("RGB")
        orig_wh = img.size
        img = img.resize((self.size, self.size), Image.BICUBIC)
        cover = torch.from_numpy(
            np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1) * 2 - 1

        if self.force_mask == "secret":
            is_secret = True
        elif self.force_mask == "normal":
            is_secret = False
        else:
            is_secret = bool(rng.random() < self.cfg.p_secret_mask)

        if is_secret:
            cell = self.secret_cell_height
            if cell is None:
                hs = self.cfg.secret_cell_heights
                cell = hs[int(rng.integers(0, len(hs)))]
            if getattr(self, "messages", None) is not None:
                # The meaningful-text path needs the same grid geometry as the
                # random-letter path, or the text experiment would quietly run
                # half-width masks against a square-cell model.
                from scripts import mask_kwargs
                dw = mask_kwargs(self.script, self.cfg)["double_width"]
                mask_pil, letters, layout, secret_text = self.messages.render(
                    self.size, cell, seed, self.cfg.font_path, double_width=dw)
            else:
                if getattr(self, "script", None):
                    from scripts import mask_kwargs
                    kw = mask_kwargs(self.script, self.cfg)
                else:
                    kw = {"font_path": self.cfg.font_path}
                mask_pil, letters, layout = word2mask(
                    image_height=self.size, word_cell_height=cell, seed=seed, **kw)
                secret_text = "".join(letters)
            m = (np.asarray(mask_pil, dtype=np.float32) > 127).astype(np.float32)
            cell_used = cell
        else:
            kind = self.normal_kind or self._pick_normal_kind(rng)
            m, inst_class, mask_kind = None, "", kind
            if kind == "instance":
                r = self._instance(img_id, orig_wh, rng)
                if r is not None:
                    m, inst_class = r
                else:
                    mask_kind = "instance_fallback"   # reported, never hidden
            if m is None:
                m = (_box_mask(self.size, rng) if kind == "box"
                     else _brush_mask(self.size, rng))
            secret_text = ""
            cell_used = 0
            layout = []

        if is_secret:
            mask_kind, inst_class = "secret", ""
        mask = torch.from_numpy(m).unsqueeze(0)

        return {
            "cover": cover,                       # [3,H,W] in [-1,1]
            "mask": mask,                         # [1,H,W] in {0,1}, 1 = inpaint
            # Ground truth for R. Identical to `mask` on secret samples; all
            # zeros on normal ones, which is exactly what the second rejection
            # term asks the decoder to produce for a payload-free image.
            "secret_gt": mask.clone() if is_secret else torch.zeros_like(mask),
            "is_secret": torch.tensor(bool(is_secret)),
            "caption": self.captions[img_id],
            "image_id": img_id,
            "secret_text": secret_text,
            "cell_height": int(cell_used),
            # Needed to score character accuracy: it says which cell holds which
            # character, and m/w span two cells, so the grid alone is not enough
            # to locate them.
            "layout": layout,
            # Which mask type this sample actually got, and for instance masks
            # the object class -- the editing experiment replaces it.
            "mask_kind": mask_kind,
            "instance_class": inst_class,
            "seed": int(seed),
        }


# ---------------------------------------------------------------------------
class OpenImagesECSNetDataset(CocoECSNetDataset):
    """Open Images V7 validation subset, for the cross-dataset test only.

    Built by prep_openimages.py (images + instance masks + annotations.json)
    and caption_openimages.py (captions.json, COCO format). Open Images ships no
    captions; they are generated once with a captioner and shared by every
    model evaluated, so the comparison stays fair.

    Inherits everything from the COCO loader except where instance masks come
    from: here they are pre-rasterised PNGs listed in annotations.json, already
    filtered to the same 2%-60% area band the COCO loader uses.
    """

    def __init__(self, spec, cfg, force_mask=None, secret_cell_height=None,
                 base_seed=None, limit=None, normal_kind=None, script=None,
                 messages=None):
        self.spec, self.cfg, self.size = spec, cfg, cfg.image_size
        # Must mirror CocoECSNetDataset: fall back to cfg.secret_script when no
        # script is passed. This subclass overrides __init__, so leaving it as
        # `self.script = script` silently gave Open Images the DEFAULT glyph
        # geometry while COCO got cfg.secret_script. Callers that pass the
        # script explicitly (evaluate.py) were unaffected; callers that rely on
        # the config (compare/compare_eval.py) produced half-width Open Images
        # masks scored against square-cell templates -- a geometry mismatch
        # that showed up as CER 0.9274, identical for all four baselines.
        self.script = script if script is not None else getattr(
            cfg, "secret_script", None)
        self.messages = messages
        self.normal_kind = normal_kind
        self.force_mask = force_mask
        self.secret_cell_height = secret_cell_height
        self.base_seed = base_seed
        self._coco = None
        self._ann_path = None

        self.img_dir = os.path.join(spec.root, spec.images_subdir)
        self.mask_dir = os.path.join(spec.root, "masks")
        for need in (self.img_dir, self.mask_dir):
            if not os.path.isdir(need):
                raise FileNotFoundError(f"{need} not found -- run prep_openimages.py")
        cap_path = os.path.join(spec.root, spec.caption_file)
        if not os.path.isfile(cap_path):
            raise FileNotFoundError(f"{cap_path} not found -- run caption_openimages.py")
        with open(cap_path) as f:
            cap_json = json.load(f)
        with open(os.path.join(spec.root, spec.ann_file)) as f:
            ann = json.load(f)

        caps = {}
        for a in sorted(cap_json["annotations"], key=lambda x: x["id"]):
            caps.setdefault(a["image_id"], a["caption"].strip())
        self.captions = caps
        self.file_names = {im["id"]: im["file_name"] for im in ann["images"]}
        self.masks = {im["id"]: im["masks"] for im in ann["images"]}
        # Label id -> readable name, for the editing experiment. Optional for
        # everything else, so a missing file only disables that experiment.
        self.class_names = {}
        cn = getattr(spec, "class_names_file", None)
        if cn and os.path.isfile(os.path.join(spec.root, cn)):
            import csv as _csv
            with open(os.path.join(spec.root, cn), newline="") as f:
                for row in _csv.reader(f):
                    if len(row) >= 2:
                        self.class_names[row[0]] = row[1]

        ids = sorted(i for i in caps if i in self.file_names and self.masks.get(i))
        random.Random(12345).shuffle(ids)
        if limit is not None:
            ids = ids[:limit]
        self.ids = ids

        cum, self._mask_kinds, self._mask_cum = 0.0, [], []
        for k, p in cfg.normal_mask_probs.items():
            cum += p
            self._mask_kinds.append(k)
            self._mask_cum.append(cum)
        self._total_p = cum

    def _instance(self, img_id, orig_wh, rng):
        names = self.masks.get(img_id) or []
        if not names:
            return None
        name = names[int(rng.integers(0, len(names)))]
        m = Image.open(os.path.join(self.mask_dir, name)).convert("L")
        m = np.array(m.resize((self.size, self.size), Image.NEAREST))
        m = (m > 127).astype(np.float32)
        if not 0.01 <= m.mean() <= 0.70:
            return None
        return m, self._label_name(name)

    def _label_name(self, mask_file):
        """Class name from an Open Images mask filename.

        Files are <ImageID>_<LabelName without slashes>_<BoxID>.png with a
        16-char ImageID and an 8-char BoxID. The label itself may contain
        underscores ("/m/0_k2"), so it is cut out by position, not by split.
        """
        stem = mask_file[:-4] if mask_file.endswith(".png") else mask_file
        mid = stem[17:-9]                       # strip "<id>_" and "_<box>"
        label = f"/{mid[0]}/{mid[1:]}" if mid else ""
        return self.class_names.get(label, "").strip().lower()


# ---------------------------------------------------------------------------
_REGISTRY = {
    "coco": CocoECSNetDataset,
    "openimages": OpenImagesECSNetDataset,
}


def build_dataset(spec, cfg, **kw) -> Dataset:
    if spec.kind not in _REGISTRY:
        raise ValueError(f"unknown dataset kind '{spec.kind}'; "
                         f"known: {sorted(_REGISTRY)}")
    return _REGISTRY[spec.kind](spec, cfg, **kw)


def collate(batch):
    """Default collation, except that the string and int fields stay as lists."""
    out = {}
    for k in batch[0]:
        v = batch[0][k]
        if torch.is_tensor(v):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test against the real COCO tree. Writes one grid per mask kind so
    # the masks can be eyeballed rather than trusted.
    import argparse

    from config import get_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./results/dataset_check")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    cfg = get_config()
    os.makedirs(args.out, exist_ok=True)

    for force, tag in [("normal", "normal"), ("secret", "secret"), (None, "mixed")]:
        ds = build_dataset(cfg.eval_datasets[0], cfg, force_mask=force,
                           base_seed=cfg.eval_seed, limit=64)
        print(f"\n[{tag}] {len(ds)} samples, images at {ds.img_dir}")
        rows = []
        for i in range(args.n):
            s = ds[i]
            cov = ((s["cover"] + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
            msk = (s["mask"][0] * 255).byte().numpy()
            gt = (s["secret_gt"][0] * 255).byte().numpy()
            rows.append(np.concatenate(
                [cov, np.stack([msk] * 3, -1), np.stack([gt] * 3, -1)], axis=1))
            print(f"  {i}: id={s['image_id']} secret={bool(s['is_secret'])} "
                  f"cell={s['cell_height']} white={s['mask'].mean():.3f} "
                  f"text='{s['secret_text'][:24]}' cap='{s['caption'][:48]}'")
        Image.fromarray(np.concatenate(rows, axis=0)).save(
            os.path.join(args.out, f"{tag}.png"))
        print(f"  -> {os.path.join(args.out, f'{tag}.png')}")

    # Determinism is a correctness property here, not a nicety: the saved
    # cover/stego pairs must be regenerable.
    a = build_dataset(cfg.eval_datasets[0], cfg, force_mask="secret",
                      base_seed=cfg.eval_seed, limit=8)[3]
    b = build_dataset(cfg.eval_datasets[0], cfg, force_mask="secret",
                      base_seed=cfg.eval_seed, limit=8)[3]
    same = torch.equal(a["mask"], b["mask"]) and a["secret_text"] == b["secret_text"]
    print(f"\ndeterministic with base_seed: {same}")