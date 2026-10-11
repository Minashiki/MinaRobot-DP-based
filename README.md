# MinaFeedingRobot

按 [重建规划](docs/feedingrobot_rebuild_plan.md)实现的原生 MuJoCo 单豆喂餐仿真。P2 双相机与 RGB 流程、P3 六维 delta 与安全控制均已验收收口。K240=K216=8已由用户锁定，相机安装已冻结，成像为240→216。P4 教师与数据采集留到下一轮。

## 环境与验证

继续使用 conda `feedingrobot`。`environment.yml` 通过 `requirements.lock.txt` 锁定运行环境，并可编辑安装当前项目。已有环境可按锁文件更新：

```bash
conda env update -n feedingrobot -f environment.yml
conda run -n feedingrobot python -m feedingrobot.scripts.validate_p0
conda run -n feedingrobot python -m feedingrobot.scripts.validate_p1
MUJOCO_GL=egl conda run -n feedingrobot python -m pytest -q
```

P0 会验证当前包的导入路径、依赖版本、HDF5 无损读写、DDIM 数值和六类随机源；P1 会验证固定旧提交与迁移场景的物理一致性、21 个边界提议、跨布局完整恢复，以及 50 个校准初态。

报告默认写入 `outputs/p0/report.json` 和 `outputs/p1/report.json`。P1 同时保存 `calibration_initial_states.json`、`proposal_xy.csv`、`settled_xy.csv`、`calibration_xy.png`，包含全部候选和拒绝原因。输出目录不纳入 Git。

## 最小物理接口

```python
from feedingrobot.sim.task import FeedingTask

task = FeedingTask()  # 中心初态冒烟检查
observation = task.reset(seed=42, split="debug", episode_id=1)
state = task.get_state()
task.step_physics()  # 0.001 s；也用于继续恢复快照中的剩余动作子步
task.set_state(state)
result = task.step_delta([0.001, 0., 0., 0., 0., 0.])  # 基座系 m/rad，50 个物理子步
# MUJOCO_GL=egl 下使用：task.observe() → 两路 RGB240 CHW float32 [0,1] + TCP xyz/wxyz
task.close()
```

`reset` 返回物理诊断快照；`observe()` 返回双路 RGB 与实测 TCP 白名单。显式检查一个提议可以向 `reset` 传入 `layout={"bowl_offset_m": [x, y], "bean_offset_m": [x, y]}`；无效提议抛出 `InvalidInitialState`，不会替换它。普通随机 reset 使用各 episode 的独立拒绝采样流。

实现规则、配置来源与验收细节见 [P0/P1 实现说明](docs/p0_p1_implementation.md)。逐文件来源、原始及当前 SHA256 见 [来源清单](third_party_manifest.json)。

## P2/P3 验证

```bash
MUJOCO_GL=egl conda run -n feedingrobot python -m pytest -q
conda run -n feedingrobot python -m feedingrobot.scripts.validate_p1 --output outputs/p1_p2_p3
conda run -n feedingrobot python -m feedingrobot.scripts.validate_p3
MUJOCO_GL=egl conda run -n feedingrobot python -m feedingrobot.scripts.validate_p2
```

P2 默认读取本机只读旧仓库 `/home/minashiki/FeedingRobot_DPRL` 的冻结 `calibration/fixed_0`。全量 RGB、原生240分割面积、216裁剪、低分位对照及遮挡区间写入 `outputs/p2`。K240=K216=8下，同一723状态的五种裁剪联合覆盖、30项关键事件检查及60组阶段证据全部通过；冻结安装后的全量复验也通过，见 [正式报告](outputs/p2/p2-calibration-v3-240/b34ed54e983161b6/report.json)。240→216采用DP Tool Hang档位，原256图、安装和FOV保持原值；旧84→76及锁定前定标包保留。本轮30项P2测试通过，P3保持既有验收报告。协议变更须重新运行全量渲染；P4前十条新教师验证示教继续按同一K8复验。运行契约与证据索引见 [P2/P3 实现说明](docs/p2_p3_implementation.md)。
