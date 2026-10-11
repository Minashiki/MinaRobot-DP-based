"""Run P0 import, source, HDF5, DDIM and independent-RNG acceptance."""

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import mujoco
import numpy as np
import torch
from packaging.requirements import Requirement

import feedingrobot
from feedingrobot.policies.scheduler import make_scheduler
from feedingrobot.random import derive_seed, numpy_generator, torch_generator
from feedingrobot.sim.model import ROOT, load_json
from feedingrobot.scripts.report import write_report


def check_sources():
    manifest = load_json("third_party_manifest.json")
    for entry in manifest["files"]:
        actual = hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest()
        assert actual == entry["sha256"], f"Source hash changed: {entry['path']}"
        if not entry["modifications"]:
            assert actual == entry["source_sha256"]
        for relative in entry["license_paths"]:
            assert (ROOT / relative).is_file(), relative
    assert len(manifest["sources"]) == 4
    assert all(len(source["commit"]) == 40 for source in manifest["sources"].values())
    return dict(files_verified=len(manifest["files"]), sources=manifest["sources"])


def check_environment():
    assert Path(feedingrobot.__file__).resolve().is_relative_to(ROOT / "src")
    versions = {}
    for module, distribution in [("torch", "torch"), ("torchvision", "torchvision"), ("mujoco", "mujoco"),
                                  ("numpy", "numpy"), ("PIL", "pillow"), ("gymnasium", "gymnasium"),
                                  ("mink", "mink"), ("diffusers", "diffusers"), ("h5py", "h5py")]:
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)
    assert versions["numpy"] == "2.2.6" and versions["h5py"] == "3.16.0"
    locked = 0
    for line in (ROOT / "requirements.lock.txt").read_text().splitlines():
        if line.startswith(("#", "--")) or not line.strip():
            continue
        requirement = Requirement(line)
        assert importlib.metadata.version(requirement.name) in requirement.specifier, line
        locked += 1
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    flags = ["TIME", "QPOS", "QVEL", "ACT", "HISTORY", "WARMSTART", "CTRL", "QFRC_APPLIED",
             "XFRC_APPLIED", "EQ_ACTIVE", "MOCAP_POS", "MOCAP_QUAT", "USERDATA", "PLUGIN"]
    integration = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    for flag in flags:
        assert integration & int(getattr(mujoco.mjtState, "mjSTATE_" + flag))
    return dict(python=sys.version, package_path=feedingrobot.__file__, versions=versions,
                locked_packages=locked, pip_check=result.stdout.strip(), cuda_available=torch.cuda.is_available(),
                integration_flags=flags, integration_mask=integration)


def check_hdf5():
    rng = numpy_generator("proposal", "p0-hdf5")
    image = rng.integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
    actions = rng.normal(size=(2, 6)).astype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.h5"
        with h5py.File(path, "w") as file:
            file.create_dataset("images", data=image, compression="gzip", shuffle=True)
            file.create_dataset("actions", data=actions)
        with h5py.File(path) as file:
            np.testing.assert_array_equal(file["images"][:], image)
            np.testing.assert_array_equal(file["actions"][:], actions)
            assert file["images"].dtype == np.uint8 and file["actions"].dtype == np.float32
    return dict(lossless_roundtrip=True, hdf5_version=h5py.version.hdf5_version)


def check_ddim():
    scheduler = make_scheduler()
    assert scheduler.timesteps.tolist() == list(range(90, -1, -10))
    config = load_json("configs/diffusion.json")
    errors = {}
    sample = torch.tensor([[-.7, .3, 1.4, -1.9]], dtype=torch.float64)
    prediction = torch.tensor([[2., -3., .2, -.4]], dtype=torch.float64)
    for timestep in (90, 40, 0):
        alpha = scheduler.alphas_cumprod[timestep]
        previous = timestep - 10
        alpha_previous = (scheduler.alphas_cumprod[previous] if previous >= 0
                          else scheduler.final_alpha_cumprod)
        epsilon = (sample - alpha.sqrt()*prediction) / (1-alpha).sqrt()
        expected = alpha_previous.sqrt()*prediction.clamp(-1, 1) + (1-alpha_previous).sqrt()*epsilon
        result = scheduler.step(prediction, timestep, sample, eta=config["eta"],
                                use_clipped_model_output=config["use_clipped_model_output"])
        torch.testing.assert_close(result.prev_sample, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(result.pred_original_sample, prediction.clamp(-1, 1), rtol=0, atol=0)
        again = scheduler.step(prediction, timestep, sample, eta=0., use_clipped_model_output=False)
        assert torch.equal(result.prev_sample, again.prev_sample)
        errors[str(timestep)] = float((result.prev_sample-expected).abs().max())
    return dict(config=config, timesteps=scheduler.timesteps.tolist(), max_errors=errors)


def check_rng():
    names = load_json("configs/protocol.json")["rng_namespaces"]
    seeds = {name: derive_seed(name, "p0-check") for name in names}
    assert len(set(seeds.values())) == 6
    for name in names:
        first, second = numpy_generator(name, "p0-check"), numpy_generator(name, "p0-check")
        np.testing.assert_array_equal(first.normal(size=16), second.normal(size=16))
        state = copy.deepcopy(first.bit_generator.state)
        expected = first.normal(size=16)
        first.bit_generator.state = state
        np.testing.assert_array_equal(first.normal(size=16), expected)
    streams = {name: numpy_generator(name, "p0-isolation") for name in names}
    expected = {name: numpy_generator(name, "p0-isolation").normal(size=8) for name in names}
    for consumed in names:
        streams[consumed].normal(size=100)
        for other in names:
            if other != consumed:
                np.testing.assert_array_equal(streams[other].normal(size=8), expected[other])
        streams = {name: numpy_generator(name, "p0-isolation") for name in names}
    episode = numpy_generator("proposal", "calibration", 2)
    numpy_generator("proposal", "calibration", 1).uniform(size=400)
    np.testing.assert_array_equal(episode.uniform(size=4),
                                  numpy_generator("proposal", "calibration", 2).uniform(size=4))
    generator = torch_generator("training", "p0-torch")
    state = generator.get_state()
    expected_torch = torch.randn(16, generator=generator)
    generator.set_state(state)
    assert torch.equal(torch.randn(16, generator=generator), expected_torch)
    python_rng = random.Random(derive_seed("training", "p0-python"))
    state = python_rng.getstate()
    value = python_rng.random()
    python_rng.setstate(state)
    assert python_rng.random() == value
    assert derive_seed("training", "worker", 0) != derive_seed("training", "worker", 1)
    assert derive_seed("validation", "window", 0) != derive_seed("validation", "window", 1)
    assert derive_seed("ddim", "validation", 0, 0) != derive_seed("ddim", "test", 0, 0)
    return dict(seeds=seeds, independent_streams=True, state_roundtrip=True,
                identities="proposal(split,episode); validation(window); ddim(purpose,initial_state,replan); training(worker)")


def validate():
    return dict(stage="P0", passed=True, environment=check_environment(), sources=check_sources(),
                hdf5=check_hdf5(), ddim=check_ddim(), rng=check_rng())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/p0/report.json")
    args = parser.parse_args()
    report = validate()
    write_report(args.output, report)
    print(f"P0 passed: {args.output}")


if __name__ == "__main__":
    main()
