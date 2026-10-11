"""Stable, named random streams; logical identities do not depend on call order."""

import hashlib
import json

import numpy as np

from feedingrobot.sim.model import load_json


def derive_seed(namespace, *identity, seed=None):
    protocol = load_json("configs/protocol.json")
    if namespace not in protocol["rng_namespaces"]:
        raise ValueError(f"Unknown random namespace: {namespace}")
    key = [protocol["version"], protocol["seed"] if seed is None else int(seed), namespace, *identity]
    encoded = json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def numpy_generator(namespace, *identity, seed=None):
    return np.random.default_rng(derive_seed(namespace, *identity, seed=seed))


def torch_generator(namespace, *identity, seed=None, device="cpu"):
    import torch

    return torch.Generator(device=device).manual_seed(derive_seed(namespace, *identity, seed=seed))
