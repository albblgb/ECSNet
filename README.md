# ECSNet — evaluation code and pre-trained weights

ECSNet turns a text-to-image inpainting model into a steganographic one: the
mask handed to the inpainting model *is* the secret, and a key-conditioned
extractor recovers it from the generated image. The cover image is never
transmitted, there is no residual to compare against, and the stego image is a
genuine inpainting result rather than a perturbed photograph.

This repository contains everything needed to **evaluate** the released model
and reproduce the ECSNet numbers of two tables in the paper:

* **Table 1** — *Comparison of normal image inpainting performance* (ECSNet vs.
  the original FLUX.1 Fill, three mask kinds, two datasets)
* **Table 2** — *Comparison of stego image quality and secret extraction
  accuracy* — the **ECSNet row only**

Not included: the training code, and the code and weights of the competing
methods in Table 2. Those are the authors' re-implementations of third-party
work and are out of scope for this release.

---

## 1. What you assemble

ECSNet is **not** a standalone network. It is the public FLUX.1-Fill-dev plus
three trained parts, released separately:

| Part | File | What it is |
|---|---|---|
| LoRA on the diffusion transformer | `pretrain/lora/` | rank-decomposition adapters, merged into the transformer at load time |
| Fine-tuned VAE decoder | `pretrain/vae_decoder.pt` | replaces the decoder half of the FLUX VAE; the encoder is untouched |
| Secret extractor *R* | `pretrain/decoder_R.pt` | key-conditioned network that reads the secret back out of the stego image |
| Secret key | `pretrain/secret_key.pt` | the single 128-d key everything was trained against |

The key is part of the model, not a password: `decoder_R.pt` was trained against
this exact vector, and extraction with any other key returns noise (that is the
WPR-1 column of Table 2). `evaluate.py` prints its fingerprint — `b099c0940af0`
— at start-up. If you see a different one, the key file is not the released one
and nothing will extract.

Download the weights (`pretrain.zip`) from the link in the paper, and unzip them
so that `pretrain/` sits next to the code, or point `ECSNET_CKPT` at wherever
you put it.

---

## 2. Install

Tested on Python 3.12, CUDA 12.8, one RTX PRO 6000 Blackwell (96 GB).

```bash
conda create -n ecsnet python=3.12 -y
conda activate ecsnet

# torch first -- pick the wheel matching your CUDA, this is the one we used
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

**GPU memory.** The full evaluation keeps FLUX's transformer, both text
encoders, the VAE and the extractor resident and peaks around 60 GB. On a
40–48 GB card add `--low_vram`, which offloads the text encoders between calls;
it is slower but produces identical numbers.

---

## 3. Where things live

Four environment variables decide every path. Unset, they default to
directories next to the code, which is the layout below.

```bash
export ECSNET_MODELS=/path/to/models    # FLUX.1-Fill-dev, clip-vit-large-patch14
export ECSNET_CKPT=/path/to/pretrain    # the released ECSNet weights
export ECSNET_DATA=/path/to/datasets    # coco2017, openimages_val
export ECSNET_RESULTS=/path/to/results  # where evaluate.py writes
```

```
ECSNet/
├── evaluate.py            the whole evaluation
├── report_tables.py       metrics.json -> the two tables
├── config.py              every hyper-parameter and path
├── models/                (you download)  FLUX.1-Fill-dev/ , clip-vit-large-patch14/
├── pretrain/              (you download)  lora/ , vae_decoder.pt , decoder_R.pt , secret_key.pt
├── data/                  (you download)  coco2017/ , openimages_val/
├── results/               (written)
└── tools/
    ├── download_models.sh
    ├── download_coco.sh
    ├── prep_openimages.py
    └── caption_openimages.py
```

---

## 4. Download the models and the data

```bash
# FLUX.1-Fill-dev (gated: accept the licence on the model page and `hf auth login` first)
# plus the CLIP used for CLIPScore
bash tools/download_models.sh

# COCO val2017 + annotations, ~1.3 GB -- all the evaluation needs
bash tools/download_coco.sh --no-train
```

Open Images is a 1000-image subset built from the V7 validation split. It is a
three-step preparation because the masks are distributed as per-letter zips:

```bash
python tools/prep_openimages.py select   # choose the images, write the download list
#   ... download the listed images and the mask zips as the script instructs ...
python tools/prep_openimages.py masks    # pull only the needed masks out of the zips
python tools/prep_openimages.py build    # rasterise, filter, write annotations.json
python tools/caption_openimages.py       # BLIP captions (Open Images ships none)
```

The selection is seeded, so the same 1000 images come out on every machine.
If you only want Table 1 and Table 2 on COCO, skip this section — `evaluate.py`
reports that Open Images is absent and carries on.

---

## 5. Reproduce the tables

```bash
# Table 1 -- normal inpainting, ECSNet
python evaluate.py --ckpt $ECSNET_CKPT --exp 1 \
    --mask_kinds instance box brush \
    --datasets coco_val2017 openimages_val \
    --n_samples 1000 --out $ECSNET_RESULTS

# Table 1 -- the same thing on the unmodified FLUX.1 Fill, for the comparison row
python evaluate.py --baseline --exp 1 \
    --mask_kinds instance box brush \
    --datasets coco_val2017 openimages_val \
    --n_samples 1000 --out $ECSNET_RESULTS

# Table 2 -- stego quality and extraction accuracy, ECSNet row
python evaluate.py --ckpt $ECSNET_CKPT --exp 2 \
    --cell_heights 64 --strengths 0.5 \
    --datasets coco_val2017 openimages_val \
    --n_samples 1000 --out $ECSNET_RESULTS

# collect them
python report_tables.py --root $ECSNET_RESULTS
```

Start with `--n_samples 20` to check the setup; it takes a couple of minutes
and writes to its own directory. At `--n_samples 1000` each mask kind on each
dataset is roughly one hour on a PRO 6000, so the three Table 1 runs plus
Table 2 come to about 8 GPU-hours per model.

`evaluate.py` refuses to write into a result directory that already exists, so
a finished run is never silently overwritten; delete it or pass a different
`--out`.

### What you should get

`report_tables.py` prints a `max drift` column: the largest absolute difference
between your run and the published ECSNet row. The evaluation is seeded (masks,
captions and the initial noise all come from `(eval_seed, index)`), so the same
card reproduces the paper exactly; a different GPU or driver moves the last
digit or two.

**Table 1, ECSNet**

| Dataset | Mask | PSNR | SSIM | LPIPS | CLIPScore |
|---|---|---|---|---|---|
| COCO | Instance Segmentation | 23.10 | 0.8358 | 0.1207 | 24.84 |
| COCO | Random Rectangular | 19.67 | 0.7747 | 0.1749 | 24.65 |
| COCO | Free-form | 25.49 | 0.8439 | 0.1004 | 24.68 |
| Open Images | Instance Segmentation | 21.48 | 0.8217 | 0.1531 | 24.11 |
| Open Images | Random Rectangular | 20.59 | 0.8154 | 0.1511 | 23.81 |
| Open Images | Free-form | 27.11 | 0.8806 | 0.0807 | 23.75 |

**Table 2, ECSNet** (cell height 64, strength 0.5, `latin_sq` secret script)

| Dataset | PSNR | SSIM | LPIPS | CLIPScore | F1 | IoU | CER | WPR-1 | WPR-2 |
|---|---|---|---|---|---|---|---|---|---|
| COCO | 28.52 | 0.8507 | 0.0947 | 24.68 | 0.9987 | 0.9975 | 0.0109 | 0.0000 | 0.0000 |
| Open Images | 30.92 | 0.8903 | 0.0736 | 23.73 | 0.9986 | 0.9973 | 0.0110 | 0.0000 | 0.0000 |

Every CER in the paper and above is raw hard template matching. No language
model, spell checker or other post-correction is applied anywhere.

---

## 6. Useful flags

| Flag | Effect |
|---|---|
| `--n_samples N` | how many images per dataset (the paper uses 1000) |
| `--low_vram` | offload the text encoders; for 40–48 GB cards |
| `--datasets coco_val2017` | run one dataset only |
| `--cell_heights 128 64 32` | capacity sweep: smaller cells carry more characters |
| `--strengths 0.3 0.5 0.7` | inpainting strength sweep |
| `--scripts latin_sq` | which alphabet the secret is drawn from (default of the paper) |
| `--no_merge_lora` | keep the LoRA as adapters instead of merging it |
| `--baseline` | the unmodified FLUX.1 Fill; no checkpoint needed |

`python evaluate.py --help` lists the rest.

---

## 7. Citation

```bibtex
@inproceedings{ecsnet,
  title     = {ECSNet: ...},
  author    = {...},
  booktitle = {...},
  year      = {2026}
}
```

FLUX.1-Fill-dev is distributed by Black Forest Labs under its own
non-commercial licence; the released ECSNet weights are derived from it and
inherit that restriction.
