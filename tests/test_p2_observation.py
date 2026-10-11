"""Policy whitelist and quaternion history survive complete task restoration."""

import numpy as np

from feedingrobot.sim.task import FeedingTask


def test_rgb_tcp_whitelist_and_quaternion_history_restore(monkeypatch):
    task = FeedingTask(cameras=False)
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[:, 128:] = 255
    monkeypatch.setattr(task, "render_rgb", lambda: {
        "agentview_image": rgb.copy(), "robot0_eye_in_hand_image": rgb.copy()})
    before = task.get_state()["physics"]
    observation = task.observe()
    assert set(observation) == {"agentview_image", "robot0_eye_in_hand_image",
                                "robot0_eef_pos", "robot0_eef_quat"}
    for name in ("agentview_image", "robot0_eye_in_hand_image"):
        assert observation[name].shape == (3, 240, 240)
        assert observation[name].dtype == np.float32
        assert observation[name].min() == 0. and observation[name].max() == 1.
    np.testing.assert_array_equal(task.get_state()["physics"], before)
    np.testing.assert_allclose(observation["robot0_eef_pos"], task.data.site_xpos[task.index.tcp], atol=1e-7)
    assert observation["robot0_eef_quat"].shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(observation["robot0_eef_quat"]), 1., atol=1e-7)
    state = task.get_state()
    task.previous_eef_quat *= -1
    reversed_history = task.observe()
    np.testing.assert_allclose(reversed_history["robot0_eef_quat"], -observation["robot0_eef_quat"])
    task.reset(layout=dict(bowl_offset_m=[.02, -.02], bean_offset_m=[0., 0.]))
    task.set_state(state)
    restored = task.observe()
    for name in ("agentview_image", "robot0_eye_in_hand_image"):
        np.testing.assert_array_equal(restored[name], observation[name])
    np.testing.assert_array_equal(restored["robot0_eef_pos"], observation["robot0_eef_pos"])
    np.testing.assert_array_equal(restored["robot0_eef_quat"], observation["robot0_eef_quat"])
