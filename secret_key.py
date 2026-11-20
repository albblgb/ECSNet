"""
secret_key.py
The single global secret key for KFSD.

Every stage -- decoder pre-training, evaluation, FLUX joint training, and
inference -- must use the identical key. Rather than each script generating its
own, they all go through get_secret_key(), which loads the key from disk if it
exists and otherwise generates it deterministically from a seed and saves it.
Once key_path exists it is the source of truth; the seed only matters for the
very first creation.
"""
import os
import hashlib

import torch


def get_secret_key(key_length=128, seed=20260921, path=None, device="cpu"):
    """
    Returns a float tensor of shape [key_length].

    If `path` exists, loads it (and checks the length). Otherwise generates a
    standard-normal key from `seed` and, if `path` is given, saves it there.
    """
    if path is not None and os.path.exists(path):
        key = torch.load(path, map_location="cpu")
        if key.shape != (key_length,):
            raise ValueError(
                f"Key at {path} has shape {tuple(key.shape)}, expected ({key_length},). "
                f"Refusing to silently use a different key -- delete the file or fix key_length.")
        return key.float().to(device)

    gen = torch.Generator().manual_seed(seed)
    key = torch.randn(key_length, generator=gen)
    if path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(key, path)
    return key.float().to(device)


def key_fingerprint(key):
    """Short hash of the key, for logs -- lets you confirm two runs used the same key
    without printing the key itself."""
    return hashlib.sha256(key.detach().cpu().float().numpy().tobytes()).hexdigest()[:12]


def random_wrong_key(key, generator=None):
    """A key drawn from the same distribution but different from `key`. Only used
    for the wrong-key diagnostic until wrong-key training is introduced."""
    return torch.randn(key.shape, generator=generator, device="cpu").to(key.device, key.dtype)
