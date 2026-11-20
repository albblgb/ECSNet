"""
evaluate.py
Experiments 1-3 for ECSNet.

  exp1  Normal inpainting:  (cover, instance mask, caption) -> image
        PSNR / SSIM / LPIPS against the cover. Run for ECSNet and for the
        unmodified FLUX.1-Fill-dev, and the difference between the two is
        experiment 3.1 (normal performance preservation).

  exp2  Steganography:      (cover, caption, secret mask) -> stego -> R
        PSNR / SSIM of the stego against the cover, plus mask F1 / IoU and CER
        under the correct key and under wrong keys. Saves the cover/stego pairs
        for the steganalysis work.

Two decisions that are easy to get wrong and that the numbers depend on:

**No latent blending.** A normal inpainting pipeline composites the model's
output back over the untouched cover outside the mask. Doing that here would be
fatal to exp2: the secret mask IS the inpainting region, so compositing would
make |stego - cover| exactly the letter shapes, and the secret would be legible
in a residual image. The whole scheme relies on regenerating the entire frame at
low strength so the perturbation is diffuse. exp1 is run the same way, without
blending, so that the ECSNet-vs-FLUX comparison is like for like.

**Fixed seeds per sample.** Masks, captions and the initial noise are all
derived from (eval_seed, index), so a rerun reproduces the same stego images.
Without that the saved cover/stego pairs could never be regenerated, and the
steganalysis experiments would be built on a set nobody can rebuild.

Usage:
    python evaluate.py --ckpt ./pretrain            # the released ECSNet weights
    python evaluate.py --baseline                 # unmodified FLUX, exp1 only
    python evaluate.py --ckpt ... --exp 2         # just the stego experiment
    python evaluate.py --ckpt ... --exp 2 --cell_heights 128 64 32   # capacity
"""
import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import flux_utils as FU
import metrics as M
from char_metrics import (build_templates, char_accuracy, format_strings,
                          build_soft_templates, char_accuracy_soft)
from scripts import SCRIPTS, get_script, mask_kwargs
from config import get_config
from dataset import build_dataset, collate
from decoder_v2 import DecoderV2, load_decoder_weights
from secret_key import get_secret_key, key_fingerprint


# ---------------------------------------------------------------------------
def save_image(x: torch.Tensor, path: str):
    """x: [3,H,W] in [-1,1], or [1,H,W] / [H,W] binary."""
    a = x.detach().float().cpu()
    if a.dim() == 3 and a.shape[0] == 1:
        a = a[0]
    # The mode argument is deprecated in Pillow 12 and removed in 13; for a
    # uint8 array Pillow infers "L" from 2 dimensions and "RGB" from 3.
    if a.dim() == 2:
        arr = (a.clamp(0, 1) * 255).byte().numpy()
    else:
        arr = ((a.clamp(-1, 1) + 1) / 2 * 255).byte().numpy().transpose(1, 2, 0)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self, cfg, ckpt: Optional[str], merge_lora: bool = True,
                 low_vram: bool = False):
        from diffusers import FluxFillPipeline

        self.cfg = cfg
        self.ckpt = ckpt
        self.low_vram = low_vram
        self.tta = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                      "float32": torch.float32}[cfg.dtype]
        self.name = "ecsnet" if ckpt else "flux_fill_original"

        print(f">>> loading FLUX.1-Fill-dev from {cfg.flux_path}")
        try:
            pipe = FluxFillPipeline.from_pretrained(cfg.flux_path, dtype=self.dtype)
        except TypeError:
            pipe = FluxFillPipeline.from_pretrained(cfg.flux_path,
                                                    torch_dtype=self.dtype)
        self.pipe = pipe
        self.transformer = pipe.transformer.to(self.device).eval()
        self.vae = pipe.vae.to(self.device).eval()
        self.scheduler = pipe.scheduler
        # The text encoders are ~9.7GB of the footprint and are used once per
        # batch for a short forward. On a card that cannot hold everything,
        # running them on the CPU costs a few seconds per batch and buys back
        # the whole 9.7GB -- a better trade than shrinking the batch to 1.
        self.text_device = "cpu" if low_vram else self.device
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.text_device)
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder_2 = pipe.text_encoder_2.to(self.text_device)
        if low_vram:
            self.text_encoder.float()
            self.text_encoder_2.float()   # CPU bf16 matmul is very slow
            print("    low_vram: text encoders on CPU (~9.7GB freed)")

        if not FU.selfcheck(pipe, self.device, cfg.image_size, verbose=False):
            raise RuntimeError("flux_utils disagrees with the installed diffusers")

        # The key must exist before the checkpoint is loaded: _load_ecsnet
        # compares the checkpoint's key fingerprint against it, and an
        # evaluation run against a different key than the one trained with
        # would report extraction numbers that mean nothing.
        self.key = get_secret_key(cfg.key_length, cfg.key_seed, cfg.key_path,
                                  device=self.device).float()
        self.key_fp = key_fingerprint(self.key)

        self.R = None
        if ckpt:
            self._load_ecsnet(ckpt, merge_lora)
        else:
            print(">>> baseline: unmodified FLUX.1-Fill-dev, no LoRA, no "
                  "fine-tuned decoder, no extractor")
            # ECSNet's fine-tuned decoder is held in fp32, so the original one
            # must be too: otherwise experiment 3.1 compares an fp32 decode
            # against a bf16 decode, and the delta mixes model change with
            # precision change. Weights unchanged, only the arithmetic.
            self.vae.decoder.to(torch.float32)
            if getattr(self.vae, "post_quant_conv", None) is not None:
                self.vae.post_quant_conv.to(torch.float32)
            print("    VAE decoder cast to fp32 to match ECSNet's precision")

        for m in (self.transformer, self.vae, self.text_encoder, self.text_encoder_2):
            m.requires_grad_(False)

        lh = cfg.image_size // 8
        self.latent_h = self.latent_w = lh
        self.image_seq_len = (lh // 2) * (lh // 2)
        self.sigmas_full = FU.inference_sigmas(
            self.scheduler, cfg.infer_steps, self.image_seq_len, self.device)
        self.latent_ids = FU.latent_image_ids(lh, lh, self.device, self.dtype)

        self.lpips = M.LPIPS(self.device)
        self.clip = M.CLIPScore(self.device, getattr(cfg, "clip_model_path", None))
        if not self.clip.ok:
            print(f"[!] CLIP score unavailable ({self.clip.why}); the column will "
                  f"be NaN. Download the model to cfg.clip_model_path to enable.")
        if not self.lpips.ok:
            print(f"[!] LPIPS unavailable ({self.lpips.why}); the column will be "
                  f"NaN.\n    Enable with: pip install lpips")

    def _load_ecsnet(self, ckpt: str, merge_lora: bool):
        from peft import PeftModel
        cfg = self.cfg

        lora_dir = os.path.join(ckpt, "lora")
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(f"no lora/ under {ckpt}")
        self.transformer = PeftModel.from_pretrained(self.transformer, lora_dir)
        if merge_lora:
            # Merging is what the paper claims is done at deployment: the result
            # is an ordinary FluxTransformer2DModel with no adapter attached and
            # no extra modules, which is the architectural-consistency claim of
            # experiment 3.2. Evaluating the merged model rather than the
            # adapter-wrapped one means the number reported is the number the
            # deployed artefact produces.
            self.transformer = self.transformer.merge_and_unload()
            print("    LoRA merged into the transformer")
        self.transformer = self.transformer.to(self.device).eval()

        vp = os.path.join(ckpt, "vae_decoder.pt")
        if os.path.isfile(vp):
            sd = torch.load(vp, map_location=self.device)
            loaded = []
            for name, state in sd.items():
                mod = getattr(self.vae, name, None)
                if mod is None:
                    print(f"[warn] checkpoint has '{name}' but the VAE does not")
                    continue
                mod.to(torch.float32).load_state_dict(state)
                loaded.append(name)
            print(f"    VAE decode side loaded: {', '.join(loaded)}")

        rp = os.path.join(ckpt, "decoder_R.pt")
        if os.path.isfile(rp):
            self.R = DecoderV2(ic=3, base_ch=48, depth=4, key_length=cfg.key_length,
                               use_key=True, key_hidden=256,
                               deep_supervision=True).to(self.device).float()
            mode = load_decoder_weights(self.R, rp, map_location=self.device)
            self.R.eval()
            print(f"    extraction decoder R loaded ({mode})")

        meta = os.path.join(ckpt, "meta.json")
        if os.path.isfile(meta):
            with open(meta) as f:
                m = json.load(f)
            print(f"    checkpoint step {m.get('step')}, key {m.get('key_fingerprint')}")
            if m.get("key_fingerprint") and m["key_fingerprint"] != self.key_fp:
                raise RuntimeError(
                    f"checkpoint key {m['key_fingerprint']} != evaluation key "
                    f"{self.key_fp}. Every extraction number would be meaningless.")

    # ---------------------------------------------------------------- sample --
    @torch.no_grad()
    def sample(self, cover, mask, captions, strength: float, generator=None):
        """Flow-matching Euler sampling, starting at sigma = `strength`.

        strength 1.0 starts from pure noise, which is ordinary inpainting.
        strength ~0.15 starts from the cover latent lightly perturbed, which is
        the stego mode: the regeneration stays close to the cover and the change
        is spread over the frame rather than confined to the mask.

        Returns the decoded image in [-1,1]. Nothing is composited back over the
        cover -- see the note at the top of this file.
        """
        cfg = self.cfg
        B = cover.shape[0]
        z0 = FU.encode_to_latent(self.vae, cover.to(self.device, self.dtype))
        zm = FU.encode_to_latent(
            self.vae, (cover * (1 - mask)).to(self.device, self.dtype))
        pe, pooled, text_ids = FU.encode_prompts(
            captions, self.tokenizer, self.text_encoder,
            self.tokenizer_2, self.text_encoder_2, self.text_device,
            torch.float32 if self.low_vram else self.dtype)
        if self.text_device != self.device:
            pe = pe.to(self.device, self.dtype)
            pooled = pooled.to(self.device, self.dtype)
            text_ids = text_ids.to(self.device, self.dtype)

        sig = self.sigmas_full
        i0 = int((sig <= strength).nonzero()[0].item()) if (sig <= strength).any() else 0
        sig = sig[i0:]
        if sig.numel() < 2:
            raise ValueError(f"strength {strength} leaves {sig.numel()} sigma "
                             f"steps; raise infer_steps or strength")

        noise = torch.randn(z0.shape, device=self.device, dtype=z0.dtype,
                            generator=generator)
        s0 = sig[0].to(z0.dtype)
        z = (1 - s0) * z0 + s0 * noise

        mask_packed = FU.pack_mask(mask.to(self.device, self.dtype))
        zm_packed = FU.pack_latents(zm)
        guidance = torch.full((B,), cfg.guidance_scale, device=self.device,
                              dtype=self.dtype)

        for i in range(sig.numel() - 1):
            t = sig[i].expand(B)
            v = self.transformer(
                hidden_states=FU.build_transformer_input(
                    FU.pack_latents(z), zm_packed, mask_packed),
                timestep=t.to(self.dtype), guidance=guidance,
                pooled_projections=pooled, encoder_hidden_states=pe,
                txt_ids=text_ids, img_ids=self.latent_ids, return_dict=False)[0]
            v = FU.unpack_latents(v, self.latent_h, self.latent_w)
            # Euler on dz/dt = v, integrating from sigma[i] down to sigma[i+1].
            z = z + (sig[i + 1] - sig[i]).to(z.dtype) * v

        return FU.decode_from_latent(self.vae, z.float())

    @torch.no_grad()
    def extract(self, image, key, tta: Optional[bool] = None):
        """R(image, key) logits, [B,1,H,W].

        With test-time augmentation, R also sees the horizontally, vertically
        and doubly flipped image; each output is flipped back and the four
        probability maps are averaged. R was trained with exactly these flips
        applied jointly to image and mask, so R(flip(x)) should equal
        flip(R(x)), and averaging cancels part of the per-pixel noise. The
        result is returned as logits so everything downstream is unchanged.
        """
        tta = self.tta if tta is None else tta
        k = key.unsqueeze(0).expand(image.shape[0], -1)
        x = image.float()
        if not tta:
            return self.R(x, k).float()
        acc = None
        for dims in ((), (-1,), (-2,), (-2, -1)):
            xi = torch.flip(x, dims) if dims else x
            pi = torch.sigmoid(self.R(xi, k).float())
            pi = torch.flip(pi, dims) if dims else pi
            acc = pi if acc is None else acc + pi
        prob = (acc / 4).clamp(1e-6, 1 - 1e-6)
        return torch.log(prob) - torch.log1p(-prob)

    # ------------------------------------------------------------------ exp1 --
    def run_exp1(self, spec, out_root: str, normal_kind: Optional[str] = None):
        """Normal inpainting. normal_kind pins the mask type ("instance",
        "box", "brush") so each gets its own row on the same images; None is
        the training mixture, kept for comparability with earlier runs."""
        cfg = self.cfg
        tag_dir = "exp1_inpainting" + (f"_{normal_kind}" if normal_kind else "")
        out = os.path.join(out_root, spec.name, tag_dir)
        vis = os.path.join(out, "visuals")
        os.makedirs(vis, exist_ok=True)

        ds = build_dataset(spec, cfg, force_mask="normal", base_seed=cfg.eval_seed,
                           limit=cfg.eval_n_samples, normal_kind=normal_kind)
        dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False,
                        collate_fn=collate, num_workers=4)
        print(f"\n=== exp1 inpainting | {self.name} | {spec.name} | "
              f"mask {normal_kind or 'mixed'} | {len(ds)} samples ===")

        rows, n_vis, t0 = [], 0, time.time()
        prompts_path = os.path.join(vis, "prompts.jsonl")
        pf = open(prompts_path, "w", encoding="utf-8")
        for bi, b in enumerate(dl):
            cover = b["cover"].to(self.device)
            mask = b["mask"].to(self.device)
            gen = torch.Generator(device=self.device).manual_seed(
                cfg.eval_seed * 7919 + bi)
            img = self.sample(cover, mask, b["caption"],
                              cfg.infer_strength_normal, gen)

            p = M.psnr(img, cover).cpu().tolist()
            s = M.ssim(img, cover).cpu().tolist()
            lp = self.lpips(img, cover).cpu().tolist()
            # CLIP score of the result against its caption, and of the cover as
            # the reference: the original image is the most caption-faithful
            # thing available, so the gap to it is the meaningful number.
            clip_s = self.clip(img, b["caption"]).cpu().tolist()
            clip_c = self.clip(cover, b["caption"]).cpu().tolist()
            for j in range(cover.shape[0]):
                rows.append({"image_id": b["image_id"][j], "psnr": p[j],
                             "ssim": s[j], "lpips": lp[j],
                             "clip_score": clip_s[j], "clip_score_cover": clip_c[j],
                             "mask_kind": b["mask_kind"][j],
                             "mask_area": float(b["mask"][j].mean())})
                if n_vis < cfg.eval_n_visual:
                    tag = f"{n_vis:04d}"
                    save_image(cover[j], os.path.join(vis, f"{tag}_cover.png"))
                    save_image(b["mask"][j], os.path.join(vis, f"{tag}_mask.png"))
                    save_image(img[j], os.path.join(vis, f"{tag}_edited.png"))
                    pf.write(json.dumps({"tag": tag, "image_id": b["image_id"][j],
                                         "prompt": b["caption"][j]}) + "\n")
                    n_vis += 1
            if bi % 10 == 0:
                done = len(rows)
                rate = (time.time() - t0) / max(done, 1)
                print(f"  {done}/{len(ds)}  PSNR {np.mean([r['psnr'] for r in rows]):.2f} "
                      f"SSIM {np.mean([r['ssim'] for r in rows]):.4f}  "
                      f"eta {(len(ds) - done) * rate / 60:.1f} min  {self._peak()}")
        pf.close()

        summary = {k: M.summarise([r[k] for r in rows])
                   for k in ("psnr", "ssim", "lpips", "clip_score", "clip_score_cover",
                             "mask_area")}
        from collections import Counter
        kinds = dict(Counter(r["mask_kind"] for r in rows))
        if kinds.get("instance_fallback"):
            print(f"  [!] {kinds['instance_fallback']} samples had no usable instance "
                  f"after rasterisation and used a brush stroke instead")
        self._write(out, rows, summary,
                    {"experiment": tag_dir, "model": self.name,
                     "dataset": spec.name, "mask_kind": normal_kind or "mixed",
                     "mask_kind_counts": kinds,
                     "strength": cfg.infer_strength_normal,
                     "steps": cfg.infer_steps, "guidance": cfg.guidance_scale,
                     "blending": "none", "n_visual_saved": n_vis})
        print(f"  -> PSNR {summary['psnr']['mean']:.2f}  "
              f"SSIM {summary['ssim']['mean']:.4f}  "
              f"LPIPS {summary['lpips']['mean']:.4f}  "
              f"CLIP {summary['clip_score']['mean']:.2f} "
              f"(cover {summary['clip_score_cover']['mean']:.2f})")
        return summary

    # ------------------------------------------------------------- editing --
    def run_edit(self, spec, out_root: str, edit_mask: str = "box"):
        """Object replacement: the instance mask's object is swapped for another
        object of the same kind ("a dog on the grass" -> "a tiger on the grass").

        Inpainting asks for plausible content; editing asks for SPECIFIC content
        while leaving everything else alone, so it is scored on both:
          clip_t_crop   CLIP(mask crop, "a photo of a <target>") -- is it there
          edit_acc      crop closer to <target> than to <source> -- success rate
          clip_dir      image change aligned with the <source>-><target> text change
          clip_t_full   CLIP(full image, edit prompt) -- the usual CLIP-T
          bg_psnr/ssim/lpips  outside the (dilated) mask -- background preserved
        """
        from edit_prompts import make_edit
        cfg = self.cfg
        if not self.clip.ok:
            print(f"[skip] editing needs CLIP ({self.clip.why})")
            return None
        # edit_mask="box" replaces the instance silhouette with its bounding box.
        # A silhouette fixes the shape of whatever is painted into it, so a dog
        # mask can hold a lion but not a tractor; with the silhouette, edit
        # success tracked shape compatibility (person->scarecrow 54-71%,
        # furniture 12%) rather than editing ability. Boxes are the usual
        # protocol for object replacement. "instance" keeps the old run.
        sub = "exp1_edit" if edit_mask == "instance" else f"exp1_edit_{edit_mask}"
        out = os.path.join(out_root, spec.name, sub)
        vis = os.path.join(out, "visuals")
        os.makedirs(vis, exist_ok=True)

        ds = build_dataset(spec, cfg, force_mask="normal", base_seed=cfg.eval_seed,
                           limit=cfg.eval_n_samples, normal_kind="instance")
        dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False,
                        collate_fn=collate, num_workers=4)
        print(f"\n=== exp1 editing ({edit_mask} masks) | {self.name} | {spec.name} | "
              f"{len(ds)} samples ===")

        rows, n_vis, n_noclass, t0 = [], 0, 0, time.time()
        vf = open(os.path.join(vis, "edits.jsonl"), "w", encoding="utf-8")
        for bi, b in enumerate(dl):
            keep = [j for j, c in enumerate(b["instance_class"]) if c]
            n_noclass += len(b["instance_class"]) - len(keep)
            if not keep:
                continue
            cover = b["cover"][keep].to(self.device)
            mask = b["mask"][keep].to(self.device)
            if edit_mask == "box":
                mask = _to_bbox(mask)
            caps = [b["caption"][j] for j in keep]
            srcs = [b["instance_class"][j] for j in keep]
            # Target drawn from the sample's own seed, so the same image always
            # gets the same edit, for every model.
            edits = [make_edit(sc, cp, np.random.default_rng(b["seed"][j] + 7))
                     for sc, cp, j in zip(srcs, caps, keep)]
            tgts = [e[0] for e in edits]
            prompts = [e[1] for e in edits]

            gen = torch.Generator(device=self.device).manual_seed(
                cfg.eval_seed * 7919 + bi)
            img = self.sample(cover, mask, prompts, cfg.infer_strength_normal, gen)

            crops = M.crop_to_mask(img, mask)
            crops0 = M.crop_to_mask(cover, mask)
            e_crop = self.clip.embed_images(crops)
            e_crop0 = self.clip.embed_images(crops0)
            e_full = self.clip.embed_images(list(img))
            e_full0 = self.clip.embed_images(list(cover))
            t_tgt = self.clip.embed_texts([f"a photo of a {t}" for t in tgts])
            t_src = self.clip.embed_texts([f"a photo of a {s_}" for s_ in srcs])
            t_prm = self.clip.embed_texts(prompts)

            clip_t_crop = 100 * (e_crop * t_tgt).sum(-1).clamp(min=0)
            edit_acc = ((e_crop * t_tgt).sum(-1) > (e_crop * t_src).sum(-1)).float()
            src_acc0 = ((e_crop0 * t_src).sum(-1) > (e_crop0 * t_tgt).sum(-1)).float()
            d_img = torch.nn.functional.normalize(e_full - e_full0, dim=-1)
            d_txt = torch.nn.functional.normalize(t_tgt - t_src, dim=-1)
            clip_dir = (d_img * d_txt).sum(-1)
            clip_t_full = 100 * (e_full * t_prm).sum(-1).clamp(min=0)
            bp, bs, bl = M.background_metrics(img, cover, mask, self.lpips)

            for n, j in enumerate(keep):
                rows.append({
                    "image_id": b["image_id"][j], "source": srcs[n], "target": tgts[n],
                    "prompt_substituted": edits[n][2],
                    "clip_t_crop": float(clip_t_crop[n]), "edit_acc": float(edit_acc[n]),
                    # Sanity check on the cover: CLIP should prefer the source
                    # there. If it often does not, the class names or crops are
                    # off, and edit_acc cannot be trusted either.
                    "source_acc_cover": float(src_acc0[n]),
                    "clip_dir": float(clip_dir[n]), "clip_t_full": float(clip_t_full[n]),
                    "bg_psnr": float(bp[n]), "bg_ssim": float(bs[n]),
                    "bg_lpips": float(bl[n]),
                })
                if n_vis < cfg.eval_n_visual:
                    t = f"{n_vis:04d}"
                    save_image(cover[n], os.path.join(vis, f"{t}_cover.png"))
                    save_image(mask[n], os.path.join(vis, f"{t}_mask.png"))
                    save_image(img[n], os.path.join(vis, f"{t}_edited.png"))
                    vf.write(json.dumps({"tag": t, "image_id": b["image_id"][j],
                                         "caption": caps[n], "source": srcs[n],
                                         "target": tgts[n], "prompt": prompts[n]}) + "\n")
                    n_vis += 1
            if bi % 10 == 0 and rows:
                done = len(rows)
                rate = (time.time() - t0) / max(done, 1)
                print(f"  {done}/{len(ds)}  edit_acc {np.mean([r['edit_acc'] for r in rows]):.3f} "
                      f"clip_t_crop {np.mean([r['clip_t_crop'] for r in rows]):.2f} "
                      f"bg_psnr {np.mean([r['bg_psnr'] for r in rows]):.2f}  "
                      f"eta {(len(ds) - done) * rate / 60:.1f} min  {self._peak()}")
        vf.close()
        if n_noclass:
            print(f"  [!] {n_noclass} samples skipped: no class name for the instance "
                  f"(Open Images needs {getattr(spec, 'class_names_file', None)})")
        if not rows:
            return None

        keys = ("clip_t_crop", "edit_acc", "clip_dir", "clip_t_full",
                "bg_psnr", "bg_ssim", "bg_lpips", "source_acc_cover", "prompt_substituted")
        summary = {k: M.summarise([float(r[k]) for r in rows]) for k in keys}
        self._write(out, rows, summary,
                    {"experiment": sub, "edit_mask": edit_mask,
                     "model": self.name, "dataset": spec.name,
                     "strength": cfg.infer_strength_normal, "steps": cfg.infer_steps,
                     "guidance": cfg.guidance_scale, "blending": "none",
                     "bg_mask_dilation_px": 8, "n_skipped_no_class": n_noclass,
                     "n_visual_saved": n_vis})
        sm = summary
        print(f"  -> edit_acc {sm['edit_acc']['mean']:.3f}  clip_t_crop "
              f"{sm['clip_t_crop']['mean']:.2f}  clip_dir {sm['clip_dir']['mean']:.4f}  "
              f"clip_t_full {sm['clip_t_full']['mean']:.2f}")
        print(f"     background: PSNR {sm['bg_psnr']['mean']:.2f}  SSIM "
              f"{sm['bg_ssim']['mean']:.4f}  LPIPS {sm['bg_lpips']['mean']:.4f}")
        print(f"     sanity: CLIP prefers the source on the cover in "
              f"{sm['source_acc_cover']['mean']:.1%} of crops; caption substituted in "
              f"{sm['prompt_substituted']['mean']:.1%}")
        return summary

    # ------------------------------------------------------------------ exp2 --
    def run_exp2(self, spec, out_root: str, cell_height: Optional[int] = None,
                 tag: str = "exp2_stego", strength: Optional[float] = None,
                 script: Optional[str] = None, messages=None):
        """Experiment 2. `script` (see scripts.py) writes the secret in another
        alphabet with the same model, untouched: the script-generalisation test.
        None keeps the training alphabet and the original directory names."""
        cfg = self.cfg
        # An explicit --scripts wins; otherwise fall back to the geometry the
        # model was TRAINED on (cfg.secret_script). Resolving it here, before
        # sc/mkw/the dataset are derived, is what keeps the mask generator, the
        # template reader and the recorded metadata from disagreeing -- reading
        # square-cell masks with half-width templates would score as recognition
        # error and look like a model regression.
        script = script or getattr(cfg, "secret_script", None)
        sc, sc_font = get_script(script, cfg)
        mkw = mask_kwargs(script, cfg)
        if self.R is None:
            print("[skip] exp2 needs the extraction decoder; the baseline has none")
            return None
        cell = cell_height or cfg.eval_secret_cell_height
        stren = cfg.strength_for(cell, strength)
        # The strength goes in the directory name: a sweep must not overwrite
        # its own earlier points, and a saved cover/stego pair is meaningless
        # without knowing which strength produced it.
        suffix = f"_cell{cell}_s{stren:g}" + ("_tta" if self.tta else "")
        out = os.path.join(out_root, spec.name, f"{tag}{suffix}")
        vis = os.path.join(out, "visuals")
        pair_c = os.path.join(out, "pairs", "cover")
        pair_s = os.path.join(out, "pairs", "stego")
        for d in (vis, pair_c, pair_s):
            os.makedirs(d, exist_ok=True)

        # Same images, captions and seeds for every script: only the glyphs
        # differ, so the scripts can be compared image by image.
        ds = build_dataset(spec, cfg, force_mask="secret", secret_cell_height=cell,
                           base_seed=cfg.eval_seed, limit=cfg.eval_n_samples,
                           script=script, messages=messages)
        if messages is not None:
            from messages import words_from_layout, blank_slots
            cand_f = open(os.path.join(out, "candidates.jsonl"), "w", encoding="utf-8")
        dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False,
                        collate_fn=collate, num_workers=4)
        print(f"\n=== exp2 stego | {self.name} | {spec.name} | cell {cell} | "
              f"strength {stren:g} | tta {self.tta} | {len(ds)} samples"
              + (f" | script {sc.name}" if script else "")
              + (f" | messages {messages.source}" if messages is not None else "") + " ===")

        # Templates are rendered exactly as word2mask renders the mask, so a
        # mismatch scores as a recognition error rather than as a rendering
        # difference.
        templates = build_templates(cell // 2, cell, mkw["font_path"], mkw["alphabet"],
                                    mkw["double_width"], mkw["font_scale"])
        soft_templates = build_soft_templates(
            cell // 2, cell, mkw["font_path"], mkw["alphabet"],
            double_width=mkw["double_width"], font_scale=mkw["font_scale"])
        bpc = sc.bits_per_char

        # Wrong keys are fixed for the whole run so the false-acceptance number
        # is measured on the same keys for every sample.
        g = torch.Generator().manual_seed(cfg.eval_seed + 31)
        wrong_keys = [torch.randn(cfg.key_length, generator=g).to(self.device)
                      for _ in range(cfg.eval_n_wrong_keys)]

        rows, n_vis, n_pairs, t0 = [], 0, 0, time.time()
        # Which characters get confused for which. The aggregate numbers cannot
        # distinguish a few systematic glyph pairs -- fixable by choosing a more
        # separable alphabet -- from confusion spread evenly over the alphabet,
        # which is not. Counting costs nothing here and decides that question.
        from collections import Counter
        confusion = Counter()
        for bi, b in enumerate(dl):
            cover = b["cover"].to(self.device)
            mask = b["mask"].to(self.device)
            secret = b["secret_gt"].to(self.device)
            gen = torch.Generator(device=self.device).manual_seed(
                cfg.eval_seed * 7919 + bi)
            stego = self.sample(cover, mask, b["caption"], stren, gen)

            p = M.psnr(stego, cover).cpu().tolist()
            s = M.ssim(stego, cover).cpu().tolist()
            lp = self.lpips(stego, cover).cpu().tolist()
            clip_s = self.clip(stego, b["caption"]).cpu().tolist()
            clip_c = self.clip(cover, b["caption"]).cpu().tolist()

            lg_ok = self.extract(stego, self.key)
            f1_ok, iou_ok = M.mask_f1_iou(lg_ok, secret)
            wh_ok = M.white_fraction(lg_ok)

            # Wrong keys, and the cover under the correct key: the two ways the
            # decoder is required to stay silent.
            f1_w, wh_w = [], []
            for wk in wrong_keys:
                lw = self.extract(stego, wk)
                a, _ = M.mask_f1_iou(lw, secret)
                f1_w.append(a.cpu().numpy())
                wh_w.append(M.white_fraction(lw).cpu().numpy())
            f1_w = np.mean(f1_w, axis=0)
            wh_w = np.mean(wh_w, axis=0)
            lg_cov = self.extract(cover, self.key)
            wh_cov = M.white_fraction(lg_cov).cpu().numpy()

            pred = (lg_ok > 0).float().cpu().numpy()
            prob = torch.sigmoid(lg_ok).cpu().numpy()
            for j in range(cover.shape[0]):
                # Hard matcher: binarised output vs ideal glyphs. Kept so the
                # numbers stay comparable with every earlier run.
                ca = char_accuracy(pred[j, 0], b["layout"][j], cell // 2, cell,
                                   templates)
                true_s, pred_s = format_strings(ca)
                # Soft matcher: probability map vs blurred, shifted glyphs.
                cs = char_accuracy_soft(prob[j, 0], b["layout"][j], cell // 2,
                                        cell, soft_templates)
                _, pred_soft = format_strings(cs)
                # Confusion is counted on the soft reader: it is the one the
                # alphabet / language-model decisions will be built on.
                for c in cs["per_cell"]:
                    confusion[(c["true"], c["pred"] or "_")] += 1
                rows.append({
                    "image_id": b["image_id"][j],
                    # Ink in the secret itself: the glyph-complexity covariate
                    # the script comparison is read against.
                    "white_gt": float(secret[j].mean()),
                    "psnr": p[j], "ssim": s[j], "lpips": lp[j],
                    "clip_score": clip_s[j], "clip_score_cover": clip_c[j],
                    "f1": float(f1_ok[j]), "iou": float(iou_ok[j]),
                    "white_correct": float(wh_ok[j]),
                    "f1_wrong": float(f1_w[j]), "white_wrong": float(wh_w[j]),
                    "white_cover": float(wh_cov[j]),
                    "char_accuracy": ca["char_accuracy"],
                    "blank_rate": ca["blank_rate"],
                    "template_iou": ca["mean_iou"],
                    "n_chars": ca["n_chars"],
                    # Aligned CER uses the cell grid, so it counts substitutions
                    # and misses without alignment ambiguity. The Levenshtein
                    # form is the figure usually quoted, kept alongside it
                    # because the two differ whenever characters go missing.
                    "cer_aligned": 1.0 - ca["char_accuracy"],
                    "cer": M.cer(true_s, pred_s),
                    "char_accuracy_soft": cs["char_accuracy"],
                    "top3_accuracy_soft": cs["topk_accuracy"],
                    "blank_rate_soft": cs["blank_rate"],
                    "cer_soft": M.cer(true_s, pred_soft),
                    # Capacity in a unit that compares alphabets of different
                    # size: correctly delivered characters x log2(alphabet).
                    "bits_sent": ca["n_chars"] * bpc,
                    "bits_correct": ca["n_correct"] * bpc,
                    "bits_correct_soft": cs["n_correct"] * bpc,
                })
                if messages is not None:
                    rows[-1].update(self._text_metrics(
                        b["secret_text"][j], b["layout"][j], ca, cs, pred[j, 0],
                        cell, b["image_id"][j], cand_f, words_from_layout, blank_slots,
                        spaces=messages.spaces))

                if n_pairs < cfg.eval_n_pairs:
                    nm = f"{n_pairs:05d}_{b['image_id'][j]}.png"
                    save_image(cover[j], os.path.join(pair_c, nm))
                    save_image(stego[j], os.path.join(pair_s, nm))
                    n_pairs += 1

                if n_vis < cfg.eval_n_visual:
                    t = f"{n_vis:04d}"
                    save_image(cover[j], os.path.join(vis, f"{t}_cover.png"))
                    save_image(mask[j], os.path.join(vis, f"{t}_secret_mask.png"))
                    save_image(stego[j], os.path.join(vis, f"{t}_stego.png"))
                    # Amplified residual: at strength 0.15 the true difference is
                    # invisible, and an invisible difference says nothing about
                    # whether letter shapes are leaking.
                    resid = ((stego[j] - cover[j]).abs() * 10).clamp(0, 1) * 2 - 1
                    save_image(resid, os.path.join(vis, f"{t}_residual_x10.png"))
                    save_image(torch.from_numpy(pred[j]),
                               os.path.join(vis, f"{t}_recovered_correct.png"))
                    lw = self.extract(stego[j:j + 1], wrong_keys[0])
                    save_image((lw[0] > 0).float().cpu(),
                               os.path.join(vis, f"{t}_recovered_wrong.png"))
                    with open(os.path.join(vis, "secrets.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "tag": t, "image_id": b["image_id"][j],
                            "prompt": b["caption"][j],
                            "secret_true": true_s, "secret_pred": pred_s,
                            "secret_pred_soft": pred_soft},
                            ensure_ascii=False) + "\n")
                    n_vis += 1

            if bi % 10 == 0:
                done = len(rows)
                rate = (time.time() - t0) / max(done, 1)
                print(f"  {done}/{len(ds)}  PSNR {np.mean([r['psnr'] for r in rows]):.2f} "
                      f"F1 {np.mean([r['f1'] for r in rows]):.4f} "
                      f"CER {np.mean([r['cer'] for r in rows]):.4f} "
                      f"soft {np.mean([r['cer_soft'] for r in rows]):.4f}  "
                      f"eta {(len(ds) - done) * rate / 60:.1f} min  {self._peak()}")

        keys = ("psnr", "ssim", "lpips", "clip_score", "clip_score_cover", "f1", "iou", "white_correct", "f1_wrong",
                "white_wrong", "white_cover", "char_accuracy", "blank_rate",
                "template_iou", "cer", "cer_aligned", "char_accuracy_soft",
                "top3_accuracy_soft", "blank_rate_soft", "cer_soft",
                "white_gt", "bits_sent", "bits_correct", "bits_correct_soft")
        if messages is not None:
            cand_f.close()
            keys += ("cer_text", "cer_text_soft", "wer_text_soft", "exact_soft",
                     "space_clean", "space_ink")
        summary = {k: M.summarise([r[k] for r in rows]) for k in keys}
        self._write(out, rows, summary,
                    {"experiment": tag, "model": self.name, "dataset": spec.name,
                     "cell_height": cell, "strength": stren,
                     "steps": cfg.infer_steps, "guidance": cfg.guidance_scale,
                     "blending": "none", "key_fingerprint": self.key_fp,
                     "n_wrong_keys": cfg.eval_n_wrong_keys,
                     "n_visual_saved": n_vis, "n_pairs_saved": n_pairs,
                     "tta": self.tta, "confusion_reader": "soft",
                     "script": sc.name, "alphabet": sc.alphabet,
                     "alphabet_size": len(sc.alphabet),
                     "bits_per_char": round(bpc, 4),
                     "font": str(sc_font) if sc_font else "default (DejaVu Sans Bold)",
                     "font_scale": mkw["font_scale"],
                     "messages": messages.source if messages is not None else "random letters"})
        if messages is not None:
            print(f"     message CER hard {summary['cer_text']['mean']:.4f}  soft "
                  f"{summary['cer_text_soft']['mean']:.4f}  WER {summary['wer_text_soft']['mean']:.4f}  "
                  f"exact {summary['exact_soft']['mean']:.3f}  blank slots clean "
                  f"{summary['space_clean']['mean']:.4f}")
            print(f"     candidates for the language-model step: {os.path.join(out, 'candidates.jsonl')}")
        print(f"  -> stego PSNR {summary['psnr']['mean']:.2f} "
              f"SSIM {summary['ssim']['mean']:.4f}")
        print(f"     correct key: F1 {summary['f1']['mean']:.4f} "
              f"IoU {summary['iou']['mean']:.4f}")
        print(f"     CER hard {summary['cer']['mean']:.4f} -> soft "
              f"{summary['cer_soft']['mean']:.4f}   (soft top-3 acc "
              f"{summary['top3_accuracy_soft']['mean']:.4f})")
        print(f"     wrong key  : F1 {summary['f1_wrong']['mean']:.4f} "
              f"white {summary['white_wrong']['mean']:.4f}")
        print(f"     cover+key  : white {summary['white_cover']['mean']:.4f}")
        print(f"     pairs saved: {n_pairs} in {os.path.join(out, 'pairs')}")
        self._write_confusion(out, confusion)
        return summary

    @staticmethod
    def _text_metrics(truth, layout, ca, cs, pred, cell, image_id, cand_f,
                      words_from_layout, blank_slots, min_p=0.02, spaces=True):
        """Message-level numbers for a meaningful secret, and the per-letter
        candidates the language-model correction (llm_correct.py) reads.

        Word boundaries are those the receiver sees on the grid: a new row or
        an empty slot. Whether the empty slots really come out empty is
        measured (space_clean), not assumed -- the model never saw an empty
        slot inside a row during training.
        """
        def cands_of(i):
            return [[c, round(p, 3)] for k, (c, p, _) in enumerate(cs["per_cell"][i]["candidates"])
                    if k == 0 or p >= min_p]

        if not spaces:
            # Letters only: the receiver gets no word boundaries; finding them
            # is the corrector's job. CER is on the letters.
            letters = truth.replace(" ", "")
            raw_hard = "".join(c["pred"] or "_" for c in ca["per_cell"])
            raw_soft = "".join(c["pred"] or "_" for c in cs["per_cell"])
            cand_f.write(json.dumps({"image_id": image_id, "truth": truth, "raw": raw_soft,
                                     "spaces": False,
                                     "letters": [cands_of(i) for i in range(len(layout))]},
                                    ensure_ascii=False) + "\n")
            nan = float("nan")
            return {"cer_text": M.cer(letters, raw_hard), "cer_text_soft": M.cer(letters, raw_soft),
                    "wer_text_soft": nan, "exact_soft": float(letters == raw_soft),
                    "space_clean": nan, "space_ink": nan}

        words = words_from_layout(layout)

        def text_of(res):
            return " ".join("".join((res["per_cell"][i]["pred"] or "_") for i in w)
                            for w in words)
        raw_hard, raw_soft = text_of(ca), text_of(cs)
        tw, sw = truth.split(), raw_soft.split()
        wer = sum(a != b for a, b in zip(tw, sw)) / max(len(tw), 1)

        cw = cell // 2
        inks = [float(pred[r * cell:(r + 1) * cell, c * cw:(c + 1) * cw].mean())
                for r, c in blank_slots(layout, pred.shape[0], cell)]
        clean = float(np.mean([v < 0.02 for v in inks])) if inks else 1.0

        rec = {"image_id": image_id, "truth": truth, "raw": raw_soft, "spaces": True, "words": []}
        for w in words:
            letters = [cands_of(i) for i in w]
            rec["words"].append({"raw": "".join(cs["per_cell"][i]["pred"] or "_" for i in w),
                                 "letters": letters})
        cand_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"cer_text": M.cer(truth, raw_hard), "cer_text_soft": M.cer(truth, raw_soft),
                "wer_text_soft": wer, "exact_soft": float(truth == raw_soft),
                "space_clean": clean, "space_ink": float(np.mean(inks)) if inks else 0.0}

    @staticmethod
    def _write_confusion(out: str, confusion):
        """Dump the glyph confusion counts and the worst pairs.

        The question this answers: is the character error concentrated in a few
        look-alike pairs, or spread across the alphabet? Concentrated error can
        be designed away by picking a more separable symbol set at a small cost
        in capacity; spread-out error cannot, and needs either lower density or
        an error-correcting code over the message.
        """
        total = sum(confusion.values())
        if not total:
            return
        wrong = {k: v for k, v in confusion.items() if k[0] != k[1]}
        n_wrong = sum(wrong.values())
        top = sorted(wrong.items(), key=lambda kv: -kv[1])[:20]

        # How concentrated is the error?
        #
        # "Share carried by the top 10 pairs" is a poor test on a 26-letter
        # alphabet: confusions form visual-similarity CLUSTERS (i/l/j/f, e/o,
        # b/h, w/m), and a cluster of k letters spreads its error over k(k-1)
        # ordered pairs, so a strongly structured failure still shows a low
        # top-10 share. Reported for reference, but the greedy drop list below
        # is what answers the design question.
        ranked = sorted(wrong.values(), reverse=True)
        share10 = float(sum(ranked[:10]) / n_wrong) if n_wrong else 0.0

        # Which letters to drop. Greedily remove the character carrying the most
        # remaining error -- counting both directions, since dropping a letter
        # removes the errors it causes AND the errors it attracts -- and record
        # the error removed against the capacity given up. This is the actual
        # trade a restricted symbol set makes, and log2 is the honest capacity
        # measure: dropping letters costs bits per cell, not cells.
        alphabet = sorted({t for t, _ in confusion} | {p for _, p in confusion
                                                       if p != "_"})
        remaining = dict(wrong)
        dropped, drop_curve = [], []
        import math
        while remaining and len(alphabet) - len(dropped) > 2:
            burden = {}
            for (t, p), v in remaining.items():
                burden[t] = burden.get(t, 0) + v
                if p != "_":
                    burden[p] = burden.get(p, 0) + v
            c = max(burden, key=burden.get)
            dropped.append(c)
            remaining = {(t, p): v for (t, p), v in remaining.items()
                         if t != c and p != c}
            removed = 1 - sum(remaining.values()) / n_wrong
            k = len(alphabet) - len(dropped)
            cap = math.log2(k) / math.log2(len(alphabet)) if k > 1 else 0.0
            drop_curve.append({"dropped": "".join(dropped), "n_dropped": len(dropped),
                               "error_removed": round(removed, 4),
                               "capacity_kept": round(cap, 4)})
            if removed > 0.95:
                break

        # Per-character accuracy, to see whether specific letters are hopeless.
        per_char = {}
        for (t, p), v in confusion.items():
            d = per_char.setdefault(t, [0, 0])
            d[1] += v
            if t == p:
                d[0] += v
        acc = {c: d[0] / d[1] for c, d in sorted(per_char.items()) if d[1]}

        with open(os.path.join(out, "confusion.json"), "w", encoding="utf-8") as f:
            json.dump({"counts": {f"{t}->{p}": v for (t, p), v in confusion.items()},
                       "n_cells": total, "n_wrong": n_wrong,
                       "error_rate": n_wrong / total,
                       "top10_share_of_error": share10,
                       "drop_curve": drop_curve,
                       "per_char_accuracy": acc}, f, indent=2, ensure_ascii=False)

        with open(os.path.join(out, "summary.md"), "a", encoding="utf-8") as f:
            f.write(f"\n## Glyph confusion\n\n{n_wrong} wrong of {total} cells "
                    f"({n_wrong / total:.4f}). The top 10 pairs account for "
                    f"{share10:.1%} of all errors.\n\n")
            f.write("| true -> predicted | count | share of error |\n|---|---|---|\n")
            for (t, p), v in top:
                f.write(f"| {t} -> {p} | {v} | {v / n_wrong:.3f} |\n")
            f.write("\n### Restricting the alphabet\n\n")
            f.write("Greedy: drop the character carrying the most remaining "
                    "error, counting both the errors it causes and those it "
                    "attracts. Capacity is log2(alphabet) relative to the full "
                    "set -- dropping letters costs bits per cell, not cells.\n\n")
            f.write("| dropped | n | error removed | capacity kept |\n|---|---|---|---|\n")
            for r in drop_curve[:12]:
                f.write(f"| {r['dropped']} | {r['n_dropped']} | "
                        f"{r['error_removed']:.3f} | {r['capacity_kept']:.3f} |\n")
            f.write("\n| char | accuracy |\n|---|---|\n")
            for c, a in sorted(acc.items(), key=lambda kv: kv[1]):
                f.write(f"| {c} | {a:.3f} |\n")

        print(f"     glyph errors: top 10 pairs carry {share10:.0%} "
              f"(a weak signal on a {len(alphabet)}-letter alphabet -- see the "
              f"drop curve)")
        print("       " + ", ".join(f"{t}->{p}:{v}" for (t, p), v in top[:6]))
        for r in drop_curve:
            if r["error_removed"] >= 0.5:
                print(f"       dropping '{r['dropped']}' ({r['n_dropped']} letters) "
                      f"removes {r['error_removed']:.0%} of the error and keeps "
                      f"{r['capacity_kept']:.0%} of the capacity")
                break

    @staticmethod
    def _peak() -> str:
        if not torch.cuda.is_available():
            return ""
        return f"peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB"

    # ----------------------------------------------------------------- write --
    @staticmethod
    def _write(out: str, rows: List[Dict], summary: Dict, meta: Dict):
        import csv
        with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "summary": summary}, f, indent=2)
        if rows:
            with open(os.path.join(out, "per_sample.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
            f.write(f"# {meta['experiment']} -- {meta['model']} -- {meta['dataset']}\n\n")
            for k, v in meta.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n| metric | mean | std | median | min | max | n |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for k, v in summary.items():
                f.write(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | "
                        f"{v['median']:.4f} | {v['min']:.4f} | {v['max']:.4f} | "
                        f"{v['n']} |\n")


# ---------------------------------------------------------------------------
def compare(a: Dict, b: Dict, name_a: str, name_b: str, path: str):
    """Experiment 3.1: ECSNet against the unmodified model, metric by metric."""
    lines = ["# ECSNet vs. the original FLUX.1 Fill (experiment 3.1)\n",
             f"| metric | {name_a} | {name_b} | delta |", "|---|---|---|---|"]
    for k in a:
        if k not in b:
            continue
        va, vb = a[k]["mean"], b[k]["mean"]
        lines.append(f"| {k} | {va:.4f} | {vb:.4f} | {va - vb:+.4f} |")
    lines.append("\nDelta is ECSNet minus the original. For PSNR and SSIM a "
                 "delta near zero is the claim; for LPIPS, lower is better, so "
                 "a negative delta favours ECSNet.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def _to_bbox(mask: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] binary masks -> the filled bounding box of each."""
    out = torch.zeros_like(mask)
    for i in range(mask.shape[0]):
        ys, xs = torch.nonzero(mask[i, 0] > 0.5, as_tuple=True)
        if ys.numel():
            out[i, 0, ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    return out


def _compare_with_baseline(ev, out_root, spec, subdir, summary):
    """Experiment 3.1: pair this result with the baseline run of the same
    experiment, on the same dataset, with the same mask type."""
    if ev.name == "flux_fill_original" or summary is None:
        return
    base_path = os.path.join(os.path.dirname(out_root), "flux_fill_original",
                             spec.name, subdir, "metrics.json")
    if not os.path.isfile(base_path):
        print(f"  [note] no baseline at {base_path}; run "
              f"`evaluate.py --baseline` with the same --exp/--mask_kinds first")
        return
    with open(base_path) as f:
        base = json.load(f)["summary"]
    k0 = next(iter(summary))
    if summary[k0]["n"] != base[k0]["n"]:
        print(f"  [!] {spec.name}/{subdir}: baseline has {base[k0]['n']} samples, "
              f"this run {summary[k0]['n']} -- the delta is only like-for-like "
              f"when n matches")
    compare(summary, base, ev.name, "flux_fill_original",
            os.path.join(out_root, spec.name, subdir, "exp3_1_vs_original.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="ECSNet checkpoint directory, e.g. checkpoints/ecsnet_final")
    ap.add_argument("--baseline", action="store_true",
                    help="evaluate the unmodified FLUX.1-Fill-dev (exp1 only)")
    ap.add_argument("--exp", nargs="+", default=["all"],
                    choices=["all", "1", "2", "edit"],
                    help="1 = inpainting, edit = object replacement, 2 = stego; "
                         "'all' runs all three. Several may be given.")
    ap.add_argument("--mask_kinds", nargs="+", default=None,
                    choices=["instance", "box", "brush", "mixed"],
                    help="exp1: one run per mask type on the same images. "
                         "Omit for the mixed run used so far.")
    ap.add_argument("--cell_heights", type=int, nargs="+", default=None,
                    help="run exp2 once per density -- this is the capacity ablation")
    ap.add_argument("--strengths", type=float, nargs="+", default=None,
                    help="override the per-cell strength. Several values sweep "
                         "the quality/recoverability trade at each density; "
                         "omit to use config.infer_strength_by_cell")
    ap.add_argument("--low_vram", action="store_true",
                    help="run the text encoders on the CPU, freeing ~9.7GB. "
                         "Costs a few seconds per batch; use on a card under 48GB "
                         "instead of dropping the batch size to 1")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--edit_mask", default="box", choices=["box", "instance"],
                    help="mask for the editing experiment: the instance's bounding "
                         "box (default) or its silhouette (the earlier protocol)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="only these eval datasets, by name, e.g. openimages_val")
    ap.add_argument("--scripts", nargs="+", default=None, choices=sorted(SCRIPTS),
                    help="exp2 only: write the secret in these alphabets with the "
                         "same model (script-generalisation test). Results go to "
                         "exp2_script_<script>_cell*_s*, separate from the main "
                         "exp2 runs, latin included, so nothing is overwritten.")
    ap.add_argument("--messages", default=None,
                    help="exp2 only: meaningful secret messages (letters without spaces, "
                         "filling the grid as in training) instead of random "
                         "letters -- 'coco' (COCO train captions) or a .txt file, one "
                         "sentence per line. Results go to exp2_text_cell*_s*, with "
                         "candidates.jsonl for llm_correct.py.")
    ap.add_argument("--message_spaces", action="store_true",
                    help="with --messages: one empty slot between words (fails: the "
                         "extractor fills empty slots; kept to reproduce that). "
                         "Results go to exp2_text_spaced_*.")
    ap.add_argument("--tta", action="store_true",
                    help="average R over the four flips; 4x the R cost, which is "
                         "small next to sampling")
    ap.add_argument("--no_merge_lora", action="store_true",
                    help="keep the adapter attached instead of merging it")
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare_with", default=None,
                    help="deprecated and ignored: the baseline is now found per "
                         "dataset under <out>/flux_fill_original/<dataset>/")
    args = ap.parse_args()

    if not args.ckpt and not args.baseline:
        ap.error("give --ckpt or --baseline")

    over = {}
    if args.n_samples:
        over["eval_n_samples"] = args.n_samples
        over["eval_n_pairs"] = min(args.n_samples, 1000)
    if args.batch_size:
        over["eval_batch_size"] = args.batch_size
    cfg = get_config(**over)

    out_root = args.out or os.path.join(cfg.results_dir, "eval")
    ev = Evaluator(cfg, None if args.baseline else args.ckpt,
                   merge_lora=not args.no_merge_lora, low_vram=args.low_vram)
    ev.tta = args.tta
    out_root = os.path.join(out_root, ev.name)
    os.makedirs(out_root, exist_ok=True)

    all_summaries = {}
    for spec in cfg.eval_datasets:
        if not os.path.isdir(spec.root):
            print(f"\n[skip] {spec.name}: {spec.root} does not exist yet")
            continue
        if args.datasets and spec.name not in args.datasets:
            continue
        exps = {"1", "2", "edit"} if "all" in args.exp else set(args.exp)
        if "1" in exps:
            kinds = args.mask_kinds or ["mixed"]
            for k in kinds:
                nk = None if k == "mixed" else k
                s1 = ev.run_exp1(spec, out_root, normal_kind=nk)
                d = "exp1_inpainting" + (f"_{nk}" if nk else "")
                all_summaries[f"{spec.name}/{d}"] = s1
                _compare_with_baseline(ev, out_root, spec, d, s1)
        if "edit" in exps:
            se = ev.run_edit(spec, out_root, edit_mask=args.edit_mask)
            if se:
                sub = "exp1_edit" if args.edit_mask == "instance" else f"exp1_edit_{args.edit_mask}"
                all_summaries[f"{spec.name}/{sub}"] = se
                _compare_with_baseline(ev, out_root, spec, sub, se)
        if "2" in exps:
            msgs = None
            if args.messages:
                from messages import MessageSampler
                msgs = MessageSampler(args.messages, cfg, spaces=args.message_spaces)
                print(f"messages: {len(msgs.sentences)} sentences from {args.messages}")
            for scr in (args.scripts or [None]):
                tag = f"exp2_script_{scr}" if scr else "exp2_stego"
                if msgs is not None:
                    if scr:
                        raise SystemExit("--messages and --scripts cannot be combined")
                    tag = "exp2_text_spaced" if args.message_spaces else "exp2_text"
                for cell in (args.cell_heights or [cfg.eval_secret_cell_height]):
                    for st in (args.strengths or [None]):
                        s2 = ev.run_exp2(spec, out_root, cell_height=cell, strength=st,
                                         tag=tag, script=scr, messages=msgs)
                        if s2:
                            key = (f"{spec.name}/{tag}_cell{cell}"
                                   f"_s{cfg.strength_for(cell, st):g}")
                            all_summaries[key] = s2

    with open(os.path.join(out_root, "all_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results under {out_root}")


if __name__ == "__main__":
    main()