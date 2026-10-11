# P0/P1 实现与验收

## 当前交付

P0 已补齐项目定义、可编辑安装、93 个包的环境锁和 h5py 3.16.0。NumPy 保持 2.2.6，MuJoCo 保持 3.14.0。验证命令必须在 `feedingrobot` 环境中运行，不能使用当前 shell 可能指向的 base Python。P0 会断言 `feedingrobot.__file__` 指向本仓库 `src/`，排除旧项目同名安装误用。

P1 保留旧 Panda 位置伺服、刚性 Spoon、静态 bowl、唯一 `bean_000`、桌面和简化接收面。工具质量 0.035 kg；盘子文件保留但默认不装配。未复制旧 twist adapter、旧 Gym、108 维观测或教师。

## 配置来源矩阵

| 配置 | 本轮值/行为 | 来源 |
| --- | --- | --- |
| Panda/Spoon/bowl/Bean 质量、惯量、摩擦、接触、安装与伺服 | 原值保留 | 旧项目 `56bd51a2c8966ab040fda60d9615cff4e9926725` |
| 物理步长、沉降阈值 | 0.001 s；5 s 上限、0.5 s 稳定窗口、1 mm/s、0.1 rad/s、0.4 mm 穿透、0.5 mm 出生净空 | 旧模型和 `acceptance.json` |
| `head_fixed` | true；保留 offset/jaw 中心、PD、bias 和执行器限界 | 用户范围要求 + 旧固定驱动语义 |
| 碗/Bean XY | 各轴 ±0.02 m 均匀提议，按几何/沉降筛选 | 用户重建规划，非论文默认值 |
| 初态几何规则 | 完整旋转椭球 XY 投影位于真实有限底盘，随后检查有限 geom 净空和沉降 | 本轮冻结的工程筛选规则，记录于 `protocol.json` |
| DDIM | sample、100 train timesteps、10 inference steps、leading、squaredcos、内部 clip | DP3 配置 + 用户要求 |
| scheduler 数值实现 | Diffusers 0.39.0，方向项使用裁剪前 x0；eta=0 | 当前锁定环境；不同于历史 0.11.1 实现 |
| seed/RNG | 根 seed 42、`p1-v1`、六类命名空间、稳定 SHA256 派生 | 用户要求与工程协议 |
| h5py | 3.16.0；NumPy 2.2.6 不升级 | P0 提前准备必需 HDF5 存储依赖 |

DP、DP3、Menagerie 的固定提交和许可证文件 hash 已记录为“引用”，本轮没有迁入其网络或相机资产。Panda 的 Apache-2.0 与餐具 EBiM LICENSE/NOTICE/Task3 说明均保留。餐具实际提取 commit 在旧项目中未核实，不能以许可核查 commit 替代。旧仓库没有根 LICENSE；未替其编造许可声明。

## 随机 reset 与几何定义

碗源 freejoint 在装配时删除。每次 reset 从原始 `body_pos/body_quat` 加当次基座系 XY 偏移，不能累计。Bean 以旧局部 reset XY 加提议偏移，再通过实际 `bowl_frame` 转到世界系；保留旧四元数。

本轮有效出生域采用明确的底盘承托条件：旋转椭球的完整 XY 投影位于真实半径 35 mm 的底盘内。投影椭圆的最远半径通过全部驻点计算，不采用 7 mm 球包络或只检查质心。随后使用 `mj_geomDistance` 检查实际有限碰撞体，仅沿 Z 增加到满足出生净空，并检查整颗椭球的底面及碗沿高度。抬高不会使 XY 越界变为有效。这是保守的底盘出生域；不声称覆盖所有可能由斜壁承托的初态。

接受分布是“均匀提议经上述几何和自然沉降筛选”，不是独立均匀分布。最多检查 100 个候选；仅 Bean 的预定义越界、净空、穿透或沉降无效会继续提议。机器人复位故障、数值 warning、力/速度故障和墙钟超时直接报错。

`proposal` 按 `(protocol, root_seed, split, episode_id)` 派生独立流。拒绝只推进当前 episode；额外验证不影响其它 episode。其它五类流在 P0 固定命名和派生规则，实际教师/采样器/训练/验证/DDIM 消费在后续阶段接入。教师当前未启用。

## 快照与恢复

`get_state()` 返回深拷贝、schema version 1 的完整状态：

- 冻结基础模型 MJB 与物理输入/协议文件的签名、MuJoCo 版本；本回合 bowl pos/quat 独立保存，签名不随回合布局变化。
- `mjSTATE_INTEGRATION`、`qacc`、`sensordata`、warnings 的 lastinfo/number。
- seed/split/episode ID、tick、external_wrench、位置保持参考及状态/故障、terminated/failure_reason。
- scenario_state 的 seed/head_phase/parameters/closure_start/future_events；P1 不启用 recover，closure_start 为 None、future_events 为空。
- contacts/applied_contacts、子步接触/腕力峰值；monitor 的 peak_n/last_peak_n/impulse_ns/over_limit_s/events/pair_peaks/pair_impulses。
- candidate_index、拒绝记录、提议 RNG 状态、reset_diagnostics、physics_timing。

当前 MuJoCo 3.14 的 integration mask 为 16383，包含 TIME/QPOS/QVEL/ACT/HISTORY/WARMSTART/CTRL/QFRC_APPLIED/XFRC_APPLIED/EQ_ACTIVE/MOCAP/USERDATA/PLUGIN。没有另设 Python time；data.time 已包含在 integration 中。

恢复顺序固定为 resetData → 恢复 bowl 模型变换 → setState → 完整 forward → 再次 setState → 恢复边界和 Python 记忆。第二次 setState 保留 forward 可能改变的积分记忆。验收监视实际调用顺序，并比较 site/geom 世界变换及原生接触；不是仅比较 body_pos。

位置保持和固定头部驱动每个子步重写控制与外力，但快照已经保存这些 MuJoCo 字段，不依赖重写来补漏。head_offset、jaw_center、速度阻尼和 bias compensation 均与旧驱动逐值对比。

physics_timing 刚恢复时逐字段相等；继续推进后仅 steps 要求相等，墙钟耗时仅检查有限非负并报告。

## 物理对照与校准验收

`tests/reference/` 保存固定旧提交的原始装配器、驱动和场景/机器人配置，属于仅供验收的参考，不接入运行控制。参考装配使用经过来源 SHA256 验证的同一资产。

原始资产 XML 的字节 hash 与上游一致；装配后的 XML 按名称/类作用域和数值规范化后逐字段比较，物理 diff 必须为空。编译后进一步比较名称/DOF 顺序、质量/惯量/关节/geom/执行器/pair/mesh/sensor 数组。维度固定为 nq=19、nv=18、ngeom=233、nmesh=206、nbody=19、nu=12。四组 qpos 的完整质量矩阵采用 rtol=1e-10、atol=1e-12，失败不会自动放宽容差。

边界矩阵为“碗 9 点 + Bean 9 点 - 共享中心 1 点 = 17”，再加四个组合极端。Bean 提议方形不是全部可接受：验收必须同时确认接受者稳定沉降和越界者被正确拒绝。A→B→A reset 检查不累积；跨布局恢复后继续 50 个相同子步，比较状态、驱动、接触与 RNG，覆盖非零外力。

校准使用独立 calibration 流的 50 个有效初态。报告全部候选、接受和沉降坐标；唯一位置以 0.1 mm 量化，二维直方图采用 5 mm 单元，并保存越界计数。其目的在于提供覆盖证据，不以低覆盖为由静默缩小范围。这批校准不是正式训练、验证或测试清单。

首次完整验收中：21 个提议有 13 个接受、8 个几何拒绝；50 个校准初态使用 98 个候选，沉降位置量化后有 50 个不同位置，XY 范围约为 24.47 × 31.99 mm；四组质量矩阵最大误差均为 0。最新实际结果以 outputs/p0/report.json 和 outputs/p1/report.json 为准。
