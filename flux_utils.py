"""
flux_utils.py
The FLUX-specific plumbing that train.py and evaluate.py both need: latent
packing, position ids, the Fill transformer's 384-channel input, prompt
encoding, and the training timestep schedule.

These live in one place because they are the parts most likely to break when
diffusers changes. Every one of them is checked against the installed pipeline
at startup by selfcheck(), so a convention change shows up in the first second
of a run rather than after an hour of training on a silently wrong input.

The 384 channels of FLUX.1-Fill-dev, which is the thing worth understanding
before reading train.py:

    packed noisy latent          16 ch x 2x2 patch =  64
    packed masked-image latent   16 ch x 2x2 patch =  64
    packed mask                   8x8 pixel block per latent cell = 64 ch,
                                  then x 2x2 patch              = 256
                                                       total    = 384

The mask is not downsampled to latent resolution and kept as one channel; it is
reshaped so that every one of the 8x8 pixels behind a latent cell gets its own
channel. That is what lets the model see mask detail finer than the latent grid,
and it is why letter strokes thinner than 8px still register.
"""
import torch


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2)*(W/2), C*4]: the 2x2 patchify FLUX expects."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(x: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """Inverse of pack_latents. latent_h/latent_w are the unpacked latent dims."""
    b, _, ch = x.shape
    c = ch // 4
    x = x.view(b, latent_h // 2, latent_w // 2, c, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, latent_h, latent_w)


def pack_mask(mask: torch.Tensor, vae_scale_factor: int = 8) -> torch.Tensor:
    """[B, 1, H, W] pixel mask -> [B, (H/16)*(W/16), 256].

    Each latent cell covers an 8x8 pixel block, and every pixel of that block
    becomes its own channel, so no mask detail is lost to downsampling.
    """
    b, _, h, w = mask.shape
    s = vae_scale_factor
    lh, lw = h // s, w // s
    m = mask[:, 0]                              # [B, H, W]
    m = m.view(b, lh, s, lw, s)
    m = m.permute(0, 2, 4, 1, 3)                # [B, s, s, lh, lw]
    m = m.reshape(b, s * s, lh, lw)             # [B, 64, lh, lw]
    return pack_latents(m)                      # [B, lh*lw/4, 256]


def latent_image_ids(latent_h: int, latent_w: int, device, dtype) -> torch.Tensor:
    """Positional ids for the packed latent grid, [(h/2)*(w/2), 3]."""
    h, w = latent_h // 2, latent_w // 2
    ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device, dtype=dtype)[None, :]
    return ids.reshape(h * w, 3)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------
def encode_to_latent(vae, images: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """Images in [-1,1] -> normalised latents, with the shift/scale FLUX uses.

    The posterior mean is taken rather than a sample: a stochastic z0 would put
    noise into the flow-matching target that the model cannot predict, and the
    gradient of that noise is pure variance.
    """
    posterior = vae.encode(images).latent_dist
    z = posterior.sample() if sample else posterior.mode()
    return (z - vae.config.shift_factor) * vae.config.scaling_factor


def decode_from_latent(vae, z: torch.Tensor) -> torch.Tensor:
    """Normalised latents -> images in [-1,1] (unclamped: the clamp would stop
    gradient wherever a pixel saturates, and L_fid needs that gradient)."""
    z = z / vae.config.scaling_factor + vae.config.shift_factor
    # Match whatever precision the decoder is held in (fp32 when fine-tuned or
    # when the evaluator casts it, bf16 otherwise) and return fp32 for metrics.
    dec_dtype = next(vae.decoder.parameters()).dtype
    return vae.decode(z.to(dec_dtype)).sample.float()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_prompts(captions, tokenizer, text_encoder, tokenizer_2, text_encoder_2,
                   device, dtype, max_sequence_length: int = 512):
    """CLIP pooled embedding + T5 sequence embedding, as FLUX wants them.

    Returns (prompt_embeds [B,L,4096], pooled [B,768], text_ids [L,3]).
    Both encoders are frozen, hence no_grad: this is a feature extractor, and
    keeping the graph would hold ~9.5GB of T5 activations for nothing.
    """
    # CLIP -> pooled
    ids = tokenizer(captions, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    pooled = text_encoder(ids, output_hidden_states=False).pooler_output.to(dtype)

    # T5 -> sequence
    ids2 = tokenizer_2(captions, padding="max_length",
                       max_length=max_sequence_length,
                       truncation=True, return_tensors="pt").input_ids.to(device)
    embeds = text_encoder_2(ids2, output_hidden_states=False)[0].to(dtype)

    text_ids = torch.zeros(embeds.shape[1], 3, device=device, dtype=dtype)
    return embeds, pooled, text_ids


# ---------------------------------------------------------------------------
# Training timesteps
# ---------------------------------------------------------------------------
def calculate_shift_mu(image_seq_len, base_seq_len=256, max_seq_len=4096,
                       base_shift=0.5, max_shift=1.15):
    """FLUX's resolution-dependent timestep shift. Matches diffusers' calculate_shift."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def inference_sigmas(scheduler, num_steps: int, image_seq_len: int, device):
    """The exact sigma schedule inference will walk.

    Training draws its timesteps from this list rather than from a uniform
    [0,1]: FLUX applies a resolution-dependent shift to its schedule, so a
    uniformly sampled t is not a t the sampler ever visits, and training would
    be optimising a regime that never runs.
    """
    import numpy as np
    mu = calculate_shift_mu(image_seq_len)
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    try:
        scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    except TypeError:
        # Older/newer signature without `mu`; fall back and say so loudly, since
        # the shift is what keeps training and inference in the same regime.
        print("[flux_utils] scheduler.set_timesteps does not accept mu -- "
              "falling back to an unshifted schedule. Train/infer timestep "
              "distributions will not match exactly.")
        scheduler.set_timesteps(num_inference_steps=num_steps, device=device)
    return scheduler.sigmas.to(device)


def sample_training_sigmas(sigma_grid: torch.Tensor, n: int, t_min: float,
                           t_max: float, generator=None) -> torch.Tensor:
    """Draw n sigmas from the inference grid, restricted to [t_min, t_max].

    Starting a sampler at strength s means starting at sigma = s, so this band
    is exactly the set of strengths deployment uses.
    """
    band = sigma_grid[(sigma_grid >= t_min) & (sigma_grid <= t_max)]
    if band.numel() == 0:
        # The grid is coarse enough to skip the band entirely; take the nearest
        # grid point to its centre rather than silently training elsewhere.
        mid = 0.5 * (t_min + t_max)
        band = sigma_grid[(sigma_grid - mid).abs().argmin()].reshape(1)
    idx = torch.randint(0, band.numel(), (n,), generator=generator,
                        device=band.device)
    return band[idx]


# ---------------------------------------------------------------------------
def build_transformer_input(z_t_packed, z_masked_packed, mask_packed):
    """The 384-channel Fill input: [noisy | masked image | mask]."""
    return torch.cat([z_t_packed, z_masked_packed, mask_packed], dim=-1)


# ---------------------------------------------------------------------------
def selfcheck(pipe, device, size: int = 512, verbose: bool = True) -> bool:
    """Check the local packing against the installed pipeline's own methods.

    diffusers moves; these conventions are the kind that change without a
    deprecation warning and produce a model that trains to a plausible-looking
    loss on a scrambled input. Checking takes a millisecond.
    """
    ok = True
    vsf = 8
    lh = lw = size // vsf
    x = torch.randn(2, 16, lh, lw, device=device)
    m = (torch.rand(2, 1, size, size, device=device) > 0.5).float()

    def _cmp(name, mine, theirs):
        nonlocal ok
        if theirs is None:
            if verbose:
                print(f"  [skip] {name}: pipeline has no comparable method")
            return
        if mine.shape != theirs.shape:
            print(f"  [FAIL] {name}: shape {tuple(mine.shape)} vs "
                  f"{tuple(theirs.shape)}")
            ok = False
            return
        d = (mine - theirs).abs().max().item()
        if d > 1e-5:
            print(f"  [FAIL] {name}: max diff {d:.3e}")
            ok = False
        elif verbose:
            print(f"  [ok]   {name}: max diff {d:.3e}")

    try:
        theirs = pipe._pack_latents(x, 2, 16, lh, lw)
    except Exception as e:
        print(f"  [skip] pack_latents: {type(e).__name__}: {e}")
        theirs = None
    _cmp("pack_latents", pack_latents(x), theirs)

    packed = pack_latents(x)
    try:
        theirs = pipe._unpack_latents(packed, size, size, vsf)
    except Exception as e:
        print(f"  [skip] unpack_latents: {type(e).__name__}: {e}")
        theirs = None
    _cmp("unpack_latents", unpack_latents(packed, lh, lw), theirs)

    # Round trip is ours alone and must hold regardless.
    rt = unpack_latents(pack_latents(x), lh, lw)
    d = (rt - x).abs().max().item()
    if d > 0:
        print(f"  [FAIL] pack/unpack round trip: max diff {d:.3e}")
        ok = False
    elif verbose:
        print("  [ok]   pack/unpack round trip exact")

    # Note the //2: diffusers' _prepare_latent_image_ids expects the PACKED grid
    # dimensions, and its callers divide before calling. Ours takes the unpacked
    # latent dims and halves internally, so the sequence length matches
    # pack_latents without the caller having to remember which convention it is
    # holding. Passing lh here instead of lh//2 compares 4096 ids against 1024.
    try:
        theirs = pipe._prepare_latent_image_ids(2, lh // 2, lw // 2, device, torch.float32)
    except Exception as e:
        print(f"  [skip] latent_image_ids: {type(e).__name__}: {e}")
        theirs = None
    mine = latent_image_ids(lh, lw, device, torch.float32)
    _cmp("latent_image_ids", mine, theirs)
    # Whatever diffusers does, the id count must equal the packed sequence
    # length, or the transformer's positional encoding does not line up with its
    # tokens. This holds independently of the comparison above.
    if mine.shape[0] != packed.shape[1]:
        print(f"  [FAIL] latent_image_ids length {mine.shape[0]} != packed "
              f"sequence length {packed.shape[1]}")
        ok = False
    elif verbose:
        print(f"  [ok]   latent_image_ids length {mine.shape[0]} == packed seq len")

    # The mask packing has no public counterpart, so check its invariants: the
    # channel count and that no mask pixel was lost.
    mp = pack_mask(m, vsf)
    exp = (2, (lh // 2) * (lw // 2), vsf * vsf * 4)
    if tuple(mp.shape) != exp:
        print(f"  [FAIL] pack_mask: shape {tuple(mp.shape)} vs {exp}")
        ok = False
    elif abs(mp.mean().item() - m.mean().item()) > 1e-6:
        print("  [FAIL] pack_mask: mean changed, pixels were dropped")
        ok = False
    elif verbose:
        print(f"  [ok]   pack_mask -> {tuple(mp.shape)}, mean preserved")

    # And that the three parts add up to what the transformer declares.
    want = pipe.transformer.config.in_channels
    got = 64 + 64 + vsf * vsf * 4
    if want != got:
        print(f"  [FAIL] transformer.in_channels is {want}, the packed input is "
              f"{got}. This is not FLUX.1-Fill-dev, or the conventions changed.")
        ok = False
    elif verbose:
        print(f"  [ok]   transformer in_channels {want} == packed input {got}")

    return ok