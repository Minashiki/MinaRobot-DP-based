"""Dual RGB, unchanged dynamics and the shared image preprocessing contract."""

import mujoco
import numpy as np
import pytest

from feedingrobot.sim.images import crop_image, image_to_cnn, resize_area, rgb240_to_observation
from feedingrobot.sim.model import ROOT, asset_files, load_json, load_model
from feedingrobot.sim.task import FeedingTask
from feedingrobot.sim.vision import VisionRenderer


@pytest.fixture(scope="module")
def camera_model():
    return load_model()


def test_two_visual_cameras_preserve_mass_and_dynamics(camera_model):
    model, index, config, _ = camera_model
    baseline, old_index, _, _ = load_model(cameras=False)
    assert model.nq == baseline.nq and model.nv == baseline.nv
    assert model.nbody == baseline.nbody+2
    camera_geoms = [g for g in range(model.ngeom)
                    if "_d435i_visual_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    assert len(camera_geoms) == 18
    assert not model.geom_contype[camera_geoms].any()
    assert not model.geom_conaffinity[camera_geoms].any()
    assert not model.body_mass[np.unique(model.geom_bodyid[camera_geoms])].any()
    for offset in [np.zeros(7), np.linspace(-.025, .025, 7),
                   np.linspace(.02, -.02, 7), np.full(7, .01)]:
        data, old_data = mujoco.MjData(model), mujoco.MjData(baseline)
        data.qpos[index.qpos] = old_data.qpos[old_index.qpos] = np.array(config["reset_q"])+offset
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(baseline, old_data)
        mass, old_mass = np.empty((model.nv, model.nv)), np.empty((baseline.nv, baseline.nv))
        mujoco.mj_fullM(model, data, mass)
        mujoco.mj_fullM(baseline, old_data, old_mass)
        np.testing.assert_array_equal(mass, old_mass)
        np.testing.assert_array_equal(data.qfrc_bias, old_data.qfrc_bias)


def test_camera_attachment_and_optical_axes(camera_model):
    model, index, config, _ = camera_model
    data = mujoco.MjData(model)
    data.qpos[index.qpos] = config["reset_q"]
    mujoco.mj_forward(model, data)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)
           for n in ("agentview", "robot0_eye_in_hand")]
    np.testing.assert_allclose(model.cam_fovy[ids], 69.)
    camera_config = load_json("configs/cameras.json")
    fixed = next(camera for camera in camera_config["cameras"] if camera["name"] == "agentview")
    body_rotation, optical_rotation = np.empty(9), np.empty(9)
    mujoco.mju_quat2Mat(body_rotation, np.array(fixed["body_quaternion_wxyz"], dtype=float))
    mujoco.mju_quat2Mat(optical_rotation, np.array(camera_config["optical_quaternion_wxyz"], dtype=float))
    body_rotation, optical_rotation = body_rotation.reshape(3, 3), optical_rotation.reshape(3, 3)
    expected_position = fixed["body_position_m"] + body_rotation @ camera_config["optical_center_m"]
    np.testing.assert_allclose(data.cam_xpos[ids[0]], expected_position)
    np.testing.assert_allclose(data.cam_xmat[ids[0]].reshape(3, 3), body_rotation @ optical_rotation)
    assert (-data.cam_xmat[ids[0]].reshape(3, 3)[:, 2])[2] < 0
    wrist = model.cam_bodyid[ids[1]]
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[wrist]) == "link7"
    before = data.cam_xpos[ids].copy()
    data.qpos[index.qpos[0]] += .1
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.cam_xpos[ids[0]], before[0])
    assert np.linalg.norm(data.cam_xpos[ids[1]]-before[1]) > .01


def test_camera_assets_registered_once_and_hashed():
    paths = asset_files()
    assert ROOT / "configs/cameras.json" in paths
    assert len([p for p in paths if p.name.startswith("d435i_")]) == 9
    assert not any("realsense_d435i" in str(p) for p in asset_files(cameras=False))
    model, _, _, _ = load_model()
    assert len([i for i in range(model.nmesh)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) or "").startswith("d435i_")]) == 9


def test_render_is_synchronous_and_does_not_mutate_physics(camera_model):
    model, index, config, _ = camera_model
    data = mujoco.MjData(model)
    data.qpos[index.qpos] = config["reset_q"]
    mujoco.mj_forward(model, data)
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(model, kind))
    after = np.empty_like(before)
    mujoco.mj_getState(model, data, before, kind)
    renderer = VisionRenderer(model)
    try:
        images = renderer.render_rgb(data)
        assert set(images) == {"agentview_image", "robot0_eye_in_hand_image"}
        for name, rgb in images.items():
            assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
        assert not renderer.scene_option.geomgroup[3:].any()
        assert not renderer.scene_option.sitegroup.any()
        mujoco.mj_getState(model, data, after, kind)
        np.testing.assert_array_equal(before, after)
    finally:
        renderer.close()


def test_exact_area_uint8_and_single_fixed_normalization():
    config = load_json("configs/cameras.json")
    assert config["render_size"] == [256, 256]
    assert config["observation_size"] == 240 and config["crop_size"] == 216
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, 128:] = 255
    small = resize_area(image)
    assert small.shape == (240, 240, 3) and small.dtype == np.uint8
    assert np.all(small[:, :120] == 0) and np.all(small[:, 120:] == 255)
    observed = rgb240_to_observation(small)
    assert observed.shape == (3, 240, 240) and observed.dtype == np.float32
    assert observed.min() == 0. and observed.max() == 1.
    cnn = image_to_cnn(small)
    assert cnn.shape == (3, 216, 216) and cnn.dtype == np.float32
    assert np.all(cnn[:, :, :108] == -1.) and np.all(cnn[:, :, 108:] == 1.)
    grey = np.full((240, 240, 3), 128, dtype=np.uint8)
    np.testing.assert_array_equal(image_to_cnn(grey),
                                  np.full((3, 216, 216), np.float32(128)/255*2-1, dtype=np.float32))
    with pytest.raises(ValueError, match="240x240 HWC uint8"):
        image_to_cnn(observed)
    with pytest.raises(ValueError, match="240x240 HWC uint8"):
        image_to_cnn(small.astype(np.float32)/255)


def test_rgb_restore_matches_after_cross_layout_reset():
    task = FeedingTask()
    try:
        state = task.get_state()
        task.render_rgb()  # Warm the OpenGL renderer before exact pixel comparison.
        reference = task.render_rgb()
        np.testing.assert_array_equal(task.get_state()["physics"], state["physics"])
        task.reset(layout={"bowl_offset_m": [.02, -.02], "bean_offset_m": [0., 0.]})
        task.render_rgb()
        task.set_state(state)
        restored = task.render_rgb()
        np.testing.assert_array_equal(task.get_state()["physics"], state["physics"])
        for name in reference:
            np.testing.assert_array_equal(restored[name], reference[name])
            np.testing.assert_array_equal(resize_area(restored[name]), resize_area(reference[name]))
            np.testing.assert_array_equal(image_to_cnn(resize_area(restored[name])),
                                          image_to_cnn(resize_area(reference[name])))
    finally:
        task.close()


def test_crop_center_training_reproducibility_and_extremes():
    image = np.arange(240*240*3).reshape(240, 240, 3)
    np.testing.assert_array_equal(crop_image(image), image[12:228, 12:228])
    for offset in [(0, 0), (0, 24), (24, 0), (24, 24)]:
        top, left = offset
        np.testing.assert_array_equal(crop_image(image, offset=offset), image[top:top+216, left:left+216])
    left, right = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(20):
        np.testing.assert_array_equal(crop_image(image, training=True, rng=left),
                                      crop_image(image, training=True, rng=right))
    with pytest.raises(ValueError, match="explicit random generator"):
        crop_image(image, training=True)
