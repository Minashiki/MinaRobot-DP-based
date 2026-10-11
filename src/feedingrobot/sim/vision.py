"""Synchronous policy RGB from fixed and rigid wrist MuJoCo cameras."""

import mujoco

from feedingrobot.sim.model import load_json, named_id


CAMERA_FIELDS = {"agentview": "agentview_image",
                 "robot0_eye_in_hand": "robot0_eye_in_hand_image"}


def policy_scene_option():
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    option.sitegroup[:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    option.frame = mujoco.mjtFrame.mjFRAME_NONE
    option.label = mujoco.mjtLabel.mjLABEL_NONE
    return option


class VisionRenderer:
    def __init__(self, model):
        for camera in CAMERA_FIELDS:
            named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        height, width = load_json("configs/cameras.json")["render_size"]
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.scene_option = policy_scene_option()

    def render_rgb(self, data):
        """Render both views without forwarding or advancing physics."""
        result = {}
        for camera, field in CAMERA_FIELDS.items():
            self.renderer.update_scene(data, camera=camera, scene_option=self.scene_option)
            result[field] = self.renderer.render().copy()
        return result

    def close(self):
        self.renderer.close()
