"""
decoder_v2.py
Secret decoder: recovers the hidden binary pattern (word mask) from a stego image,
conditioned on the secret key.

Architecture: residual U-Net, self-attention at the bottleneck, optional deep
supervision. Unchanged from the probe version except for how the key enters.

Key conditioning is FiLM, not spatial concatenation:

    h  ->  h * (1 + d_gamma(key)) + beta(key)      (per channel, spatially uniform)

applied after every encoder level, the bottleneck, and every decoder level.

Why FiLM rather than the earlier concatenation:
  * Warm start. Concatenation changes every block's input channel count, so a
    checkpoint trained without a key can no longer be loaded. FiLM leaves every
    conv shape untouched; the key-free probe checkpoint loads as-is and only the
    FiLM modules are new.
  * Identity at init. The FiLM output heads are zero-initialised, so d_gamma = 0
    and beta = 0 and the network computes exactly what the warm-start checkpoint
    computed. Nothing learned is lost at the moment the key is switched on.
  * Right inductive bias. The key is a global secret with no spatial meaning.
    The old key map (Linear -> repeat along rows) imposed a row-by-row pattern
    unrelated to image content; FiLM is per-channel and spatially uniform.
  * Suits wrong-key rejection later. "Wrong key -> all black" is multiplicative
    gating, and the (1 + d_gamma) term can drive features toward zero directly.

With use_key=False no key modules are created, and the state_dict is identical to
the probe version -- old key-free checkpoints load strictly.

Final layer emits raw logits. Apply sigmoid in the loss (BCEWithLogitsLoss) or
explicitly at inference, never inside the model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels, max_groups=8):
    """GroupNorm needs an exact divisor. GroupNorm rather than BatchNorm because
    batch statistics are unreliable at the small batch sizes this trains at."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.norm1 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class SelfAttention(nn.Module):
    """Single-head self-attention over spatial positions. Bottleneck only, where
    the map is small enough for N^2 attention."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(_gn_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(B, 3, C, H * W).unbind(1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class KeyFiLM(nn.Module):
    """
    key [B, key_length] -> shared MLP embedding -> one (d_gamma, beta) pair per
    conditioned level.

    Output heads are zero-initialised: at init every level receives
    d_gamma = 0, beta = 0, i.e. an exact identity transform.
    """
    def __init__(self, key_length, level_channels, hidden=256):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(key_length, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 2 * c) for c in level_channels])
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, key):
        e = self.embed(key)
        return [h(e).chunk(2, dim=1) for h in self.heads]


def _film(x, params):
    d_gamma, beta = params
    return x * (1 + d_gamma[:, :, None, None]) + beta[:, :, None, None]


class DecoderV2(nn.Module):
    def __init__(self, ic=3, base_ch=48, depth=4, key_length=128, use_key=True,
                 key_hidden=256, attn_heads_at_bottleneck=True, deep_supervision=True,
                 key_channels=None):
        """
        base_ch: width at full resolution; doubles per downsample, capped at base_ch*8
        depth:   number of downsamples (512x512, depth 4 -> 32x32 bottleneck)
        use_key: FiLM key conditioning on/off. Off gives the exact probe architecture.
        key_channels: accepted and ignored, for compatibility with older call sites
                      written for the concatenation version.
        """
        super().__init__()
        self.depth = depth
        self.use_key = use_key
        self.deep_supervision = deep_supervision

        chs = [min(base_ch * (2 ** i), base_ch * 8) for i in range(depth + 1)]
        self.chs = chs

        self.stem = nn.Sequential(
            nn.Conv2d(ic, chs[0], 3, 1, 1, bias=False),
            nn.GroupNorm(_gn_groups(chs[0]), chs[0]),
            nn.SiLU(inplace=True),
        )

        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(depth):
            self.enc_blocks.append(nn.Sequential(ResBlock(chs[i], chs[i]), ResBlock(chs[i], chs[i])))
            self.downsamples.append(nn.Conv2d(chs[i], chs[i + 1], 3, 2, 1))

        self.bottleneck = nn.Sequential(
            ResBlock(chs[depth], chs[depth]),
            SelfAttention(chs[depth]) if attn_heads_at_bottleneck else nn.Identity(),
            ResBlock(chs[depth], chs[depth]),
        )

        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.aux_heads = nn.ModuleList() if deep_supervision else None
        for i in reversed(range(depth)):
            self.upsamples.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 4, 2, 1))
            self.dec_blocks.append(nn.Sequential(ResBlock(chs[i] * 2, chs[i]), ResBlock(chs[i], chs[i])))
            if deep_supervision:
                self.aux_heads.append(nn.Conv2d(chs[i], 1, 1))

        self.final = nn.Conv2d(chs[0], 1, 1)  # raw logits

        if use_key:
            # conditioned levels, in forward order: encoder 0..depth-1,
            # bottleneck, decoder depth-1..0
            film_channels = ([chs[i] for i in range(depth)] + [chs[depth]]
                             + [chs[i] for i in reversed(range(depth))])
            self.key_film = KeyFiLM(key_length, film_channels, hidden=key_hidden)

    def forward(self, image, key=None, return_aux=False):
        film = None
        if self.use_key:
            if key is None:
                raise ValueError("use_key=True but no key was passed")
            film = self.key_film(key)
        li = 0  # index into film params, advances in forward order

        x = self.stem(image)

        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            if film is not None:
                x = _film(x, film[li]); li += 1
            skips.append(x)
            x = self.downsamples[i](x)

        x = self.bottleneck(x)
        if film is not None:
            x = _film(x, film[li]); li += 1

        aux_logits = []
        for j, i in enumerate(reversed(range(self.depth))):
            x = self.upsamples[j](x)
            skip = skips[i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec_blocks[j](torch.cat([x, skip], dim=1))
            if film is not None:
                x = _film(x, film[li]); li += 1
            if self.deep_supervision and return_aux:
                aux_logits.append(self.aux_heads[j](x))

        logits = self.final(x)
        if return_aux and self.deep_supervision:
            return logits, aux_logits
        return logits


def load_decoder_weights(model, ckpt_path, map_location="cpu"):
    """
    Load a checkpoint into DecoderV2, handling the key-free -> key-conditioned case.

      * same architecture (keyed ckpt into keyed model, or key-free into key-free):
        loads strictly.
      * key-free ckpt into a keyed model: every non-key weight loads; the FiLM
        modules keep their zero init, so the model behaves exactly like the
        key-free checkpoint until training moves them.

    Anything else -- mismatched base_ch/depth, or an old concatenation-style keyed
    checkpoint -- raises with an explanation rather than half-loading silently.

    Returns "strict" or "keyfree_into_keyed".
    """
    sd = torch.load(ckpt_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    if any(k.startswith("key_encoder.") for k in sd):
        raise RuntimeError(
            f"{ckpt_path} is from the old concatenation-style key decoder "
            f"(has key_encoder.*). Its block shapes differ from this FiLM version; "
            f"load it with decoder_v2_concat_legacy.py instead.")

    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"Shape mismatch loading {ckpt_path}: base_ch/depth probably differ from the "
            f"checkpoint. Inspect it with:\n"
            f"  sd=torch.load(path); print(sd['stem.0.weight'].shape[0], "
            f"len({{k.split('.')[1] for k in sd if k.startswith('enc_blocks.')}}))\n"
            f"Original error: {e}") from e

    film_missing = [k for k in missing if k.startswith("key_film.")]
    other_missing = [k for k in missing if not k.startswith("key_film.")]
    if other_missing or unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint {ckpt_path}:\n"
            f"  missing (non-key): {other_missing[:8]}{' ...' if len(other_missing) > 8 else ''}\n"
            f"  unexpected: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

    return "keyfree_into_keyed" if film_missing else "strict"


def deep_supervision_loss(logits, aux_logits, target, criterion, aux_weight=0.4):
    """Main loss plus down-weighted auxiliary losses at each intermediate scale."""
    loss = criterion(logits, target)
    for aux in aux_logits:
        t = target if aux.shape[-2:] == target.shape[-2:] else \
            F.interpolate(target, size=aux.shape[-2:], mode="nearest")
        loss = loss + aux_weight * criterion(aux, t) / max(len(aux_logits), 1)
    return loss


if __name__ == "__main__":
    import tempfile, os
    B, H = 2, 256
    img = torch.randn(B, 3, H, H)
    key = torch.randn(B, 128)

    # 1. key-free model == probe architecture; save it as a stand-in for best.pt
    m0 = DecoderV2(base_ch=16, depth=3, use_key=False).eval()
    path = os.path.join(tempfile.mkdtemp(), "keyfree.pt")
    torch.save(m0.state_dict(), path)

    # 2. warm-start a keyed model from it
    m1 = DecoderV2(base_ch=16, depth=3, use_key=True).eval()
    mode = load_decoder_weights(m1, path)

    with torch.no_grad():
        a = m0(img)
        b = m1(img, key)
    print(f"load mode: {mode}")
    print(f"max |keyed - keyfree| at init: {(a - b).abs().max().item():.2e}  (expect 0)")

    out, aux = m1(img, key, return_aux=True)
    print(f"out {tuple(out.shape)}, aux scales {[tuple(x.shape[-2:]) for x in aux]}")
    print(f"params keyed {sum(p.numel() for p in m1.parameters())/1e6:.2f}M, "
          f"FiLM {sum(p.numel() for p in m1.key_film.parameters())/1e6:.3f}M")
