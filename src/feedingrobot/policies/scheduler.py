"""The frozen x0/DDIM configuration; no policy network is introduced in P0."""

from diffusers import DDIMScheduler

from feedingrobot.sim.model import load_json


def make_scheduler():
    config = load_json("configs/diffusion.json")
    scheduler = DDIMScheduler(**{key: value for key, value in config.items()
                                 if key not in {"num_inference_steps", "eta", "use_clipped_model_output"}})
    scheduler.set_timesteps(config["num_inference_steps"])
    return scheduler
