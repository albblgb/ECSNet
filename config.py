"""
config.py
All hyperparameters for ECSNet on FLUX.1-Fill-dev.

Reference hardware: one RTX PRO 6000 Blackwell (96 GB). Evaluation also fits in
48 GB with --low_vram, and the 1000-sample runs in the paper used the former.

Paths are resolved from ECSNET_MODELS / ECSNET_CKPT / ECSNET_DATA /
ECSNET_RESULTS (see below); nothing here is machine-specific.
Stack:  torch 2.8.0+cu128, diffusers 0.41.0, transformers 5.20.2, peft 0.21.2.

Everything the experiment plan asks to be tunable lives here, in particular the
six loss weights of Eq. (total). Nothing in train.py or evaluate.py hard-codes a
path or a weight; they take a Config.

Flow matching, not noise prediction:
  z_t = (1-t) z_0 + t eps,   v = eps - z_0,   z0_hat = z_t - t * v_hat
so the "noise" objectives are velocity objectives, and the one-step estimate is
a subtraction rather than the DDPM alpha/sigma algebra.

Timesteps are drawn from the low-noise band that inference actually uses
(strength ~0.1-0.2). Training across the full [0,1] range while only ever
sampling near t=0 at deployment sends most of the gradient to a regime the model
never runs in.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_WARNED = set()   # one extrapolation note per strength, not one per call

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------
# Every path below is resolved from an environment variable with a default
# inside the release directory, so the code runs unmodified after
#
#   export ECSNET_MODELS=/somewhere/models      # FLUX.1-Fill-dev, CLIP
#   export ECSNET_CKPT=/somewhere/pretrain      # the released ECSNet weights
#   export ECSNET_DATA=/somewhere/datasets      # coco2017, openimages_val
#
# Unset, they point at models/, pretrain/ and data/ next to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("ECSNET_MODELS", os.path.join(_HERE, "models"))
CKPT_DIR = os.environ.get("ECSNET_CKPT", os.path.join(_HERE, "pretrain"))
DATA_DIR = os.environ.get("ECSNET_DATA", os.path.join(_HERE, "data"))
RESULTS_DIR = os.environ.get("ECSNET_RESULTS", os.path.join(_HERE, "results"))


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Open Images is downloaded (prepared under datasets/ecsnet/openimages_val) and
# is evaluation-only -- it never participates in training. Adding another set is
# one entry here plus one loader branch in dataset.py -- no change to train.py or
# evaluate.py, both of which iterate over whatever is listed.
@dataclass
class DatasetSpec:
    name: str
    kind: str                      # "coco" | "openimages"
    root: str
    split: str                     # "train2017" / "val2017" / ...
    # Captions: COCO ships them. Open Images does not, so they must be
    # generated; caption_file then points at the generated jsonl.
    caption_file: Optional[str] = None
    # Instance-segmentation source for the normal (editing) masks.
    ann_file: Optional[str] = None
    # Images live at <root>/<images_subdir>/<split>. COCO on this machine is the
    # nested layout, so images_subdir is "images"; a flat checkout uses "".
    images_subdir: str = "images"
    in_training: bool = False
    # Open Images only: label id -> readable class name, used by the editing
    # experiment to name the object being replaced.
    class_names_file: Optional[str] = None


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # The original FLUX.1-Fill-dev, downloaded separately (tools/download_models.sh).
    # ECSNet is this model plus the released LoRA, VAE decoder and extractor.
    flux_path: str = os.path.join(MODELS_DIR, "FLUX.1-Fill-dev")

    code_dir: str = _HERE
    results_dir: str = RESULTS_DIR
    # Directory holding the released weights: lora/, vae_decoder.pt,
    # decoder_R.pt, meta.json, secret_key.pt.
    checkpoint_dir: str = CKPT_DIR

    # Unused in the evaluation code path; kept so Config stays one object.
    prompt_cache_dir: Optional[str] = None

    # Training-only warm start for the extraction decoder R. Not needed to
    # evaluate the released model, whose decoder_R.pt is already trained.
    init_decoder_ckpt: Optional[str] = None

    # Full CLIP (with vision tower) for the CLIP-score metric. FLUX ships only
    # CLIP's text encoder, so this is a separate download. None disables it.
    clip_model_path: Optional[str] = os.path.join(MODELS_DIR, "clip-vit-large-patch14")

    # Font for the secret masks. None falls back to DejaVuSans-Bold.
    font_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    image_size: int = 512
    # Per-sample throughput is identical at batch 2 and batch 4 (0.78 s/sample
    # measured both ways), so the GPU is already saturated at batch 2 and a
    # larger batch buys no speed -- only steadier gradients. Batch 4 peaks at
    # 61.3GB of 95GB; batch 8 extrapolates to ~89GB, too close to the limit to
    # be worth it for a 26-hour run.
    batch_size: int = 4
    grad_accum: int = 2            # effective batch 8
    num_workers: int = 8

    # Fraction of training samples carrying a secret mask rather than a normal
    # one. The two mask distributions are very different, so L_edit and L_steg
    # are computed on disjoint halves of the stream and monitored separately.
    p_secret_mask: float = 0.5

    # Secret mask densities, sampled uniformly. Smaller cells = more characters
    # = higher capacity and a harder extraction problem.
    secret_cell_heights: List[int] = field(default_factory=lambda: [128, 64, 32])

    # Writing system for the TRAINING secret masks. None = "latin", the a-z set
    # with the original m/w double-width rule. "latin_sq" is the same 26 letters
    # with EVERY character given two cell slots, i.e. a square glyph box.
    #
    # Why square -- and what it does NOT fix. font_size is int(cell_height *
    # 0.8 * font_scale), i.e. it depends on the cell HEIGHT only, so a wider
    # slot does not enlarge the glyph; it adds horizontal padding. The measured
    # ink widths at cell 64, in latent columns after the VAE's 8x downsample
    # (a half-width slot is 4 columns, a square slot 8):
    #       l,i 2.1 | a 4.2 | o 4.4 | b 4.6 | m 6.6
    # So at half width MOST letters overflow their own slot and neighbouring
    # glyphs overlap in latent space. That, not pixel count, is why 'm' 0.991
    # and 'w' 0.935 lead the table: they already took two slots and so fit with
    # margin. Square cells give every letter that margin, and the gain is the
    # removal of inter-character crosstalk.
    #
    # It therefore does NOT help the largest single confusion: l->i 74, i->l 52
    # and j->l 21 (11% of all errors) are a vertical/dot resolution problem, and
    # l and i are 2.1 columns wide -- they never overflowed. Only a larger
    # cell_height adds glyph pixels. Hence the expectation for square cells
    # alone is CER ~0.03-0.05, not 0.02; reaching 0.02 likely needs cell 128.
    # secret_cell_heights still samples 128/64/32, so one run covers all three
    # and the operating point is chosen at evaluation time.
    #
    # The alphabet is deliberately NOT restricted: dropping the 12 most
    # confusable letters would remove 90.8% of the error at 81% of the capacity,
    # but those letters include e/a/r/n/s/l, so meaningful English text becomes
    # unwritable and the text-message experiment dies with it.
    #
    # Capacity cost at cell 64: 119 -> 64 characters per image.
    secret_script: Optional[str] = "latin_sq"

    # Normal mask shapes for the editing branch. Instance polygons are the real
    # thing; the other two keep the branch alive when an annotation is missing
    # or degenerate, and widen the shape distribution.
    normal_mask_probs: Dict[str, float] = field(default_factory=lambda: {
        "instance": 0.70,
        "box": 0.15,
        "brush": 0.15,
    })

    max_train_samples: Optional[int] = None

    # ------------------------------------------------------------------
    # Key
    # ------------------------------------------------------------------
    # One fixed global key, generated deterministically from key_seed on first
    # use and written to key_path, which is then the source of truth -- training,
    # evaluation and inference must all see the identical vector.
    key_length: int = 128
    key_seed: int = 20260921
    # Ships with the released weights. It is NOT regenerated from key_seed here:
    # decoder_R.pt was trained against this exact vector (fingerprint
    # b099c0940af0, printed at start-up), and a different key extracts nothing.
    key_path: str = os.path.join(CKPT_DIR, "secret_key.pt")

    # ------------------------------------------------------------------
    # Flow matching / sampling regime
    # ------------------------------------------------------------------
    # The band of t training draws from, matched to the inference strength:
    # strength s starts the sampler at t = s, so this covers 0.1-0.5.
    #
    # Widened from [0.05, 0.25] on 2026-11-11. Higher strength measurably lowers
    # CER -- 0.212 at s0.2, 0.145 at s0.3, 0.091 at s0.5 -- but those were all
    # EXTRAPOLATION past the old t_max, so the band now covers them. 0.5 is the
    # ceiling for two reasons: beyond it L_steg becomes ill-posed (it asks for an
    # exact reconstruction of the cover latent from less than half a signal), and
    # the glyph shapes risk becoming visible in the stego image, which would cost
    # the camouflage claim the whole method rests on.
    #
    # Density check, because a wider band spreads the same budget thinner:
    #   old  6000 steps / 0.20 wide = 30,000 steps per unit t
    #   new 10000 steps / 0.40 wide = 25,000 steps per unit t   (83% of old)
    # At t_max 0.6 it would be 67%, which is too thin to justify the headroom.
    t_min: float = 0.1
    t_max: float = 0.5

    # FLUX takes guidance as a conditioning input (distilled), not via CFG.
    guidance_scale: float = 30.0

    # Training draws its timesteps from the shifted schedule FLUX actually
    # walks, not from a uniform [0,1]. That schedule is built at this
    # resolution: a 30-step grid only has ~6 points inside [t_min, t_max], and
    # training on six discrete timesteps is a much narrower problem than the
    # continuous one. 1000 makes the band effectively continuous while keeping
    # the shift identical to inference.
    train_sigma_steps: int = 1000

    # Inference settings for evaluate.py and the periodic training samples.
    infer_steps: int = 30
    # Stego branch. Strength is how far the sampler starts from the cover, so
    # it is the dial that trades image quality against how much signal the
    # perturbation can carry: higher strength means a larger, more recoverable
    # change and a lower PSNR.
    #
    # Denser text needs more signal per unit area, so the right strength is not
    # one number. infer_strength_by_cell overrides the default per density; a
    # cell height not listed falls back to infer_strength_secret.
    #
    # HARD CONSTRAINT: training drew its timesteps from [t_min, t_max] =
    # [0.05, 0.25]. A strength outside that band puts inference in a regime the
    # model never trained on, and the one-step/multi-step agreement measured so
    # far says nothing about it. strength_for() clamps and warns rather than
    # letting a sweep wander out of distribution silently.
    infer_strength_secret: float = 0.15
    infer_strength_by_cell: Dict[int, float] = field(default_factory=lambda: {
        128: 0.15,   # CER 0.106 at this setting -- already the usable point
        64: 0.20,
        32: 0.35,    # above the trained band -- see infer_strength_cap
    })
    # Inference-only ceiling, kept separate from t_max on purpose: t_max also
    # sets the TRAINING timestep band, so raising it to allow strength 0.35 at
    # evaluation would silently change any later training run (e.g. the
    # training-strategy ablation) and make it incomparable with this model.
    # Strengths in (t_max, infer_strength_cap] are allowed but are an
    # extrapolation beyond what the model was trained on; say so in the paper.
    # infer_strength_cap: float = 0.35
    infer_strength_cap: float = 1.0
    infer_strength_normal: float = 1.0    # editing branch: a real inpaint

    # ------------------------------------------------------------------
    # The six loss weights -- Eq. (total)
    # ------------------------------------------------------------------
    # L = l_edit*L_edit + l_steg*L_steg + l_rec*L_rec
    #   + l_fid*L_fid  + l_ext*L_ext  + l_rej*L_rej
    #
    # The terms are not on a common scale: velocity MSE sits around 0.1-0.5, the
    # pixel L2 terms around 0.01-0.05, BCE around 0.1-0.7. The values below put
    # the weighted terms within about an order of magnitude of each other at the
    # start of stage B. train.py logs every term separately -- retune from those
    # numbers rather than from intuition about the ratios.
    # Set from `python train.py --calibrate`, which measures each objective's
    # GRADIENT norm per parameter group rather than its loss value. Measured at
    # lambda = 1 on 4 batches:
    #
    #   term    loss     |g| T    |g| D    |g| R
    #   edit  0.30707   0.2073   0.0000   0.0000
    #   steg  0.69549   0.2386   0.0000   0.0000
    #   rec   0.00245   0.0000   0.0515   0.0000
    #   fid   0.00540   0.0000   0.0691   0.0000
    #   ext   0.40813   1.7663  14.8866   4.1288
    #   rej   0.00000   0.0000   0.0000   0.0000
    #
    # The thing the loss values hid: L_ext's pull on D is 290x L_rec's. Raising
    # lambda_rec to 50 and lambda_fid to 20 still left the anchor and the
    # counterweight overwhelmed, because the real problem was that lambda_ext
    # itself was ~7x too high.
    #
    # L_ext's gradient is also distributed very unevenly -- nearly all on D,
    # much less on T -- so equalising its TOTAL norm would starve T, and L_ext
    # is the only path by which T learns to shape a recoverable perturbation.
    # Hence the groups are settled in order: T fixes lambda_ext by matching
    # L_steg's pull there, and D's two terms then match whatever L_ext exerts
    # on D. That gives an even three-way split of gradient in both groups.
    # 2026-11-12 retrain: lambda_ext raised to 0.75 with lambda_rec/fid at 5.
    # Set by the user as a deliberate quality-for-accuracy trade, together with
    # the argument that an init-time gradient measurement is a poor basis for
    # choosing weights at all -- see the reasoning note after the table.
    #
    # Re-measured on the NEW band [0.1, 0.5] by job 1734352 (--calibrate, 4
    # batches, unit weights):
    #
    #   term     loss      |g| T    |g| D    |g| R
    #   edit  0.21107    0.2184   0.0000   0.0000
    #   steg  0.31076    0.1903   0.0000   0.0000
    #   rec   0.00241    0.0000   0.0428   0.0000
    #   fid   0.00700    0.0000   0.0873   0.0000
    #   ext   0.59013    1.1746   8.9580   3.7147
    #   rej   0.00000    0.0000   0.0000   0.0000
    #
    # Widening the band moved both sides toward balance on its own: L_ext's pull
    # on D fell 14.89 -> 8.96 while L_fid's rose 0.069 -> 0.087.
    #
    # D (VAE decoder) group, where L_ext and the two anchors compete directly:
    #
    #   rec/fid      rec              fid               ext          ext share
    #    39/29   39*0.0428=1.669  29*0.0873=2.532  0.135*8.958=1.209    22%
    #    20/20   20*0.0428=0.856  20*0.0873=1.746  0.25 *8.958=2.240    46%
    #    10/10   10*0.0428=0.428  10*0.0873=0.873  0.5  *8.958=4.479    77%
    #      5/5    5*0.0428=0.214   5*0.0873=0.437  0.75 *8.958=6.719    91%  <-
    #
    # On T: edit 0.2184 (17%), steg 0.1903 (15%), ext 0.75*1.1746=0.881 (68%).
    #
    # WHY THE 77% IS NOT THE OBJECTION IT LOOKS LIKE. These are shares at
    # INITIALISATION. They do not hold through training, and the direction they
    # move is knowable: a share is proportional to lambda * |grad|, and |grad|
    # falls as its own loss falls. L_ext starts at 0.590 and has the most room
    # to fall of any term here; L_rec (0.0024) and L_fid (0.0070) are already
    # near their floors and cannot shrink much further. So L_ext's share DECAYS
    # over training and the anchors' shares RISE -- 77% is the opening tilt, not
    # the steady state. Weights picked to equalise the opening state are
    # therefore mis-set for every step after it, which is the user's point, and
    # it is correct.
    #
    # Two structural facts make the opening tilt safer than the number reads:
    # stage A (0..2000) runs with L_ext switched off entirely, so D is shaped by
    # rec+fid alone before L_ext ever touches it; and stage_b_ramp_steps eases
    # lambda_ext in over the next 1000 steps rather than applying it as a step.
    #
    # The residual risk is real and is NOT measured by CER: D is the VAE
    # decoder, so over-tilting it costs stego PSNR, worsens the VAE round-trip
    # (COCO LPIPS 0.019 -> 0.050 already) and can make the glyph traces VISIBLE,
    # which would forfeit the camouflage claim. 10/10 at the old band was 76%
    # and the note above records that regime as having failed -- but it failed
    # with lambda_ext at 0.135 and no square cells, so it is not the same
    # experiment. Guards during the run: stego PSNR and the residual_x10 samples
    # in stage B. Guard after it: the exp1 group of slurm_jobs/smoke_v2.sbatch,
    # which compares normal-inpainting PSNR against the paper's Table 1 -- D
    # being dragged too far shows up THERE before it shows up in experiment 2.
    lambda_edit: float = 1.0
    lambda_steg: float = 1.0
    lambda_rec: float = 5.0
    lambda_fid: float = 5.0
    lambda_ext: float = 0.75
    # Could not be calibrated while satisfied -- both terms measured exactly
    # zero gradient -- so this was set from the spikes instead. At lambda 1.0,
    # the steps where rej2 became nonzero (0.038-0.074) drove R's gradient norm
    # to 5.4-7.9 against the 0.2-0.4 that L_ext contributes there: a 20x
    # imbalance that clipped R's whole update, L_ext included, by about 4x on
    # those steps. At 0.3 an activation lands at 1.5-2.4, which corrects
    # promptly without flattening everything else in the group.
    lambda_rej: float = 0.3

    # Within L_rej, the split between its two terms. Term 1 is R(I_s, K_err)->0,
    # satisfiable only through the key pathway; term 2 is R(I_c, K)->0,
    # satisfiable only through the image pathway. Decoder pre-training measured
    # a 0.1975 white fraction on clean covers, so term 2 has real work to do and
    # carries the larger share.
    rej_wrong_key_weight: float = 0.5
    rej_cover_weight: float = 1.0
    # Which parameters of R each rejection term updates. The two terms are not
    # symmetric in the network: term 1 shows R identical pixels under a
    # different key, so only FiLM can separate the cases; term 2 shows an
    # identical key with different pixels, so only the backbone can. Pointing a
    # term at the pathway that cannot satisfy it leaves one gradient direction
    # available -- suppress everything -- which takes correct-key extraction
    # down with it. That is what collapsed the decoder during pre-training.
    rej_wrong_grad: str = "film"       # "film" | "all"
    rej_cover_grad: str = "backbone"   # "backbone" | "all"
    # Deadband. BCE against an all-zero target never saturates: a pixel already
    # rejected at logit -6 still attracts gradient toward -12, and that
    # unbounded pressure leaks into L_ext. Pixels below this logit count as
    # done. rej_loss="bce" disables it, for the ablation.
    rej_loss: str = "masked"       # "masked" | "bce"
    rej_margin: float = -2.0

    # Auxiliary deep-supervision heads inside R, applied to L_ext only.
    aux_weight: float = 0.4

    # ------------------------------------------------------------------
    # Two-stage schedule
    # ------------------------------------------------------------------
    # Stage A keeps the generator honest before anything asks it to hide data:
    # only edit / steg / rec are active, so T learns to fill both mask
    # distributions and D stays a faithful autoencoder. Stage B switches on the
    # three objectives that need the one-step estimate.
    #
    # single_stage=True collapses stage A to zero steps -- that is the
    # Single-stage Joint Training arm of the training-strategy ablation.
    # Wall time depends only on total samples, since batch size does not change
    # per-sample throughput. 15000 steps x effective batch 8 = 120k samples,
    # about one epoch of COCO train2017 (118k images), at ~6.2 s/step -> ~26
    # hours. Two epochs would be 52 hours for a second pass over data the model
    # has already seen, with LoRA on a strong pretrained generator and R already
    # pre-trained; one epoch first, and extend by resuming if the curves are
    # still moving.
    single_stage: bool = False
    stage_a_steps: int = 2000
    total_steps: int = 10000

    # Ramp for the stage-B objectives, from the start of stage B. Going straight
    # to full weight is what collapsed the decoder during its pre-training.
    stage_b_ramp_steps: int = 1000

    # ------------------------------------------------------------------
    # What is trainable
    # ------------------------------------------------------------------
    # Frozen: VAE encoder E, both text encoders. Trained: LoRA on transformer T,
    # the full VAE decoder D, and the extraction decoder R.
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # Attention projections only. Keeping the MLPs out holds the adapter small,
    # which matters for the camouflage claim: after merging, the weights must
    # stay close to the originals, and that is measured in experiment 3.3.
    lora_targets: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    ])

    train_vae_decoder: bool = True
    train_extraction_decoder: bool = True

    lr_lora: float = 1e-4
    lr_vae_decoder: float = 1e-5
    lr_decoder_r: float = 1e-5     # R arrives pre-trained; it is being adapted,
                                   # not learned, so this sits well below the
                                   # 3e-4 used during its own pre-training
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    # The smoke run clipped at gradient norms of 3-6 once the stage-B terms came
    # on, i.e. the update was being scaled down 3-6x every step. Clipping that
    # hard turns the learning rate into something the schedule no longer
    # controls, and it does not scale the parameter groups evenly. 2.0 still
    # catches spikes while leaving ordinary steps alone. Watch the `gn` column:
    # if it sits far above this, the weights above are out of balance rather
    # than the clip being too tight.
    max_grad_norm: float = 2.0

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    seed: int = 42

    log_every: int = 20
    sample_every: int = 1000
    ckpt_every: int = 1000         # a 26-hour run should not lose more than
                                   # ~1.7 hours to an interruption
    keep_last_n_ckpt: int = 3

    # ------------------------------------------------------------------
    # Evaluation (experiments 1-3)
    # ------------------------------------------------------------------
    eval_n_samples: int = 1000     # per dataset, per experiment
    eval_n_visual: int = 100       # image groups saved for the paper figures
    eval_n_pairs: int = 1000       # cover/stego pairs saved for steganalysis
    eval_batch_size: int = 4
    eval_seed: int = 1234
    eval_n_wrong_keys: int = 3     # wrong keys for the false-acceptance number

    # Secret density used for the headline stego table. The capacity ablation
    # sweeps secret_cell_heights instead.
    eval_secret_cell_height: int = 64

    eval_datasets: List[DatasetSpec] = field(default_factory=lambda: [
        DatasetSpec(
            name="coco_val2017",
            kind="coco",
            root=os.path.join(DATA_DIR, "coco2017"),
            split="val2017",
            caption_file="annotations/captions_val2017.json",
            ann_file="annotations/instances_val2017.json",
            images_subdir="images",
            in_training=False,
        ),
        # Cross-dataset test, never trained on. Built by prep_openimages.py and
        # caption_openimages.py. evaluate.py skips it, with a message, until the
        # directory exists.
        DatasetSpec(
            name="openimages_val",
            kind="openimages",
            root=os.path.join(DATA_DIR, "openimages_val"),
            split="",
            caption_file="captions.json",
            ann_file="annotations.json",
            images_subdir="images",
            class_names_file="class-descriptions-boxable.csv",
            in_training=False,
        ),
    ])

    train_dataset: DatasetSpec = field(default_factory=lambda: DatasetSpec(
        name="coco_train2017",
        kind="coco",
        root=os.path.join(DATA_DIR, "coco2017"),
        split="train2017",
        caption_file="annotations/captions_train2017.json",
        ann_file="annotations/instances_train2017.json",
        images_subdir="images",
        in_training=True,
    ))

    # ------------------------------------------------------------------
    def loss_weights(self, step: int) -> Dict[str, float]:
        """The six weights in force at `step`, after stage gating and ramping.

        Stage A zeroes the three objectives that depend on the one-step
        estimate; stage B ramps them in linearly. Returned as a dict so train.py
        logs exactly what it used rather than recomputing the schedule.
        """
        base = {
            "edit": self.lambda_edit,
            "steg": self.lambda_steg,
            "rec": self.lambda_rec,
            "fid": self.lambda_fid,
            "ext": self.lambda_ext,
            "rej": self.lambda_rej,
        }
        a_steps = 0 if self.single_stage else self.stage_a_steps
        if step < a_steps:
            base["fid"] = base["ext"] = base["rej"] = 0.0
            return base
        if self.stage_b_ramp_steps > 0:
            frac = min(1.0, (step - a_steps + 1) / self.stage_b_ramp_steps)
            for k in ("fid", "ext", "rej"):
                base[k] *= frac
        return base

    def strength_for(self, cell_height: int, override: Optional[float] = None) -> float:
        """Sampling strength for a secret of this density.

        `override` wins when given, so a sweep can pass an explicit value. The
        result is clamped to the band training actually covered; going outside
        it is not a mild extrapolation, it is a regime the transformer never saw
        a gradient in.
        """
        s = override if override is not None else self.infer_strength_by_cell.get(
            cell_height, self.infer_strength_secret)
        lo, hi = self.t_min, max(self.t_max, self.infer_strength_cap)
        if s < lo or s > hi:
            c = min(max(s, lo), hi)
            print(f"[config] strength {s} is outside [{lo}, {hi}]; clamping to {c}.")
            s = c
        if s > self.t_max:
            key = ("extrap", round(s, 4))
            if key not in _WARNED:
                _WARNED.add(key)
                print(f"[config] strength {s} is above the trained band "
                      f"[{self.t_min}, {self.t_max}]: an extrapolation, allowed "
                      f"up to infer_strength_cap={self.infer_strength_cap}.")
        return float(s)

    def stage_of(self, step: int) -> str:
        return "A" if (not self.single_stage and step < self.stage_a_steps) else "B"


def get_config(**overrides) -> Config:
    """Config with keyword overrides, so a sweep or an ablation arm is one call."""
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Config has no field '{k}'")
        setattr(cfg, k, v)
    return cfg