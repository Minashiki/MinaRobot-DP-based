# MinaFeedingRobot

按 [重建规划](docs/feedingrobot_rebuild_plan.md)实现的原生 MuJoCo 单豆喂餐仿真。当前完成 P0 环境/来源与 P1 物理基线；尚未接入双相机、delta 动作、教师、策略或训练。

## 环境与验证

继续使用 conda `feedingrobot`。`environment.yml` 通过 `requirements.lock.txt` 锁定运行环境，并可编辑安装当前项目。已有环境可按锁文件更新：

```bash
conda env update -n feedingrobot -f environment.yml
conda run -n feedingrobot python -m feedingrobot.scripts.validate_p0
conda run -n feedingrobot python -m feedingrobot.scripts.validate_p1
conda run -n feedingrobot python -m pytest -q
```

P0 会验证当前包的导入路径、依赖版本、HDF5 无损读写、DDIM 数值和六类随机源；P1 会验证固定旧提交与迁移场景的物理一致性、21 个边界提议、跨布局完整恢复，以及 50 个校准初态。

报告默认写入 `outputs/p0/report.json` 和 `outputs/p1/report.json`。P1 同时保存 `calibration_initial_states.json`、`proposal_xy.csv`、`settled_xy.csv`、`calibration_xy.png`，包含全部候选和拒绝原因。输出目录不纳入 Git。

## 最小物理接口

```python
from feedingrobot.sim.task import FeedingTask

task = FeedingTask()  # 中心初态冒烟检查
observation = task.reset(seed=42, split="debug", episode_id=1)
state = task.get_state()
task.step_physics()  # 0.001 s，位置伺服保持参考；尚无策略动作接口
task.set_state(state)
```

`observation` 是物理诊断快照；后续视觉策略的输入白名单在相应阶段实现。显式检查一个提议可以向 `reset` 传入 `layout={"bowl_offset_m": [x, y], "bean_offset_m": [x, y]}`；无效提议抛出 `InvalidInitialState`，不会替换它。普通随机 reset 使用各 episode 的独立拒绝采样流。

实现规则、配置来源与验收细节见 [P0/P1 实现说明](docs/p0_p1_implementation.md)。逐文件来源、原始及当前 SHA256 见 [来源清单](third_party_manifest.json)。
