# FeedingRobot 纯仿真重建规划

日期：2026-10-11。本文只规划重建，不执行资产迁移、依赖安装、数据采集或训练。

## 1. 实现依据与最终范围

`refer/refer.md` 指定了两篇文献：

1. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion（DP）**：采用其双视角视觉条件、CNN 编码、Time-series Diffusion Transformer、动作扩散训练和滚动执行方法。
2. **3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations（DP3）**：采用其动作/状态 LinearNormalizer 数值规则、`sample` 参数化和 DDIM 配置。

实现优先级为：**本次已确认的用户要求 > 对应方法的官方实现 > 旧项目实现**。论文、官方代码默认值和 FeedingRobot 工程约定分别记录，不能混称为论文参数。

DP3 的核心是点云表示与点编码器，官方主干为条件 U-Net。本项目明确使用 RGB、CNN 和 Transformer，因此应称为“采用 DP 视觉 Transformer 方法、DP3 归一化与采样配置的 FeedingRobot”，不能宣称完整复现 DP3 的三维感知方法或其泛化成绩。[S1][S2][S3][S4]

### 已确认的配置

| 项目 | 本次实现 |
| --- | --- |
| 环境 | 继续使用 conda `feedingrobot` |
| 仿真 | 原生 MuJoCo，Gymnasium 只封装 `reset/step`；使用 MuJoCo 光栅化 RGB 离屏渲染 |
| 机械臂 | 旧项目 Panda，保留位置伺服及物理参数 |
| 任务资产 | 固定 Spoon、单颗 `bean_000`、回合内固定的 bowl；plate 文件保留，默认场景不装配 |
| 初始状态分布 | 首轮随机化碗基座系 XY 与 Bean 碗内局部 XY，提议范围各轴 `±2 cm`，经预定义几何/沉降检查后冻结 |
| 相机外观 | 两个 Menagerie D435i 网格实例，只作视觉附件、关闭碰撞 |
| 真正的成像元件 | 两个独立 MuJoCo `<camera>`：固定俯视 `agentview`、腕部 `robot0_eye_in_hand` |
| 相机模态 | 仅 RGB，不使用深度、点云、分割图或物体真值作为策略输入 |
| 水平视场 | D435 RGB 模组近似 `hfov=69°`；本项目不是硬件光学标定复现 |
| 图像流程 | `256×256 uint8` → 面积重采样 `240×240 uint8` → `/255` → 训练随机裁剪 / 推理中心裁剪 `216×216` → `2x−1`（DP Tool Hang 档位） |
| 图像数值 | `uint8 / 255 → float32 [0,1] → 2x-1 → CNN 输入 [-1,1]`，对齐 DP 固定图像 normalizer |
| 数据增强 | 只有上述训练随机裁剪 |
| 动作 | **6 维**：基座系 `Δxyz(3) + Δ轴角(3)`；勺子刚性固定，不保留第七维 |
| 状态归一化 | 本体状态逐维 min/max 仿射变换至 `[-1,1]`，近常值维度只平移 |
| 动作归一化 | 同一 LinearNormalizer 数值规则；推理结束后反归一化回米/弧度 |
| 后置处理 | 不加动作 clip、tanh、范数限幅、平滑或额外比例缩放 |
| 控制频率 | `20 Hz`，每条 delta 动作持续 `0.05 s` |
| 物理频率 | 延续旧模型 `1000 Hz`，每条动作推进 50 个 `0.001 s` 物理子步 |
| 窗口 | `horizon=16`、`T_o=2`、`T_a=8`，每 `0.4 s` 仿真时间重规划一次 |
| 网络输出 | 干净的归一化动作序列 `x̂₀`，`prediction_type='sample'` |
| 损失 | 对 `x̂₀` 与干净动作 `x₀` 计算 MSE，不回归噪声 epsilon |
| 扩散时间网格 | 训练 100 个 timestep；推理 DDIM 10 次更新；`timestep_spacing='leading'` |
| 底层控制 | 复用旧 Mink IK 核心，新增 delta 位姿目标接口 |
| 安全层 | 保留力保护与碰撞监控，触发急停并终止回合，不修正策略动作 |

任务终点暂沿用旧项目的简化嘴部/接收面：舀起单 Bean、携带、交付、撤离。此接收目标属于 FeedingRobot 任务定义，不是新增的人体模拟课题。首轮复用旧资产与任务判据，但将碗位置和 Bean 落点随机化；能力声明限定在冻结的小范围初始分布内。本轮不加入 RL、UR5e、多豆或新的可动勺具。

## 2. 已核查的基线与复用边界

### 来源固定

| 来源 | 本次核查提交 | 用途 |
| --- | --- | --- |
| `Minashiki/Diffusion-based-Feeding-Robot` | `56bd51a2c8966ab040fda60d9615cff4e9926725` | 物理资产、装配、IK 核心、任务判据、教师几何路径 |
| `real-stanford/diffusion_policy` | `5ba07ac6661db573af695b419a7947ecb704690f` | CNN、Transformer、时序窗口与训练基线 |
| `YanjieZe/3D-Diffusion-Policy` | `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97` | LinearNormalizer、`sample` loss、DDIM 配置 |
| `google-deepmind/mujoco_menagerie` | `0059d4335f8156206f63a35662313385f7ad6d74` | D435i 网格、材质与许可证 |

实际导入时记录 URL、commit、逐文件 SHA256、许可证和局部修改，之后不从浮动 `main` 自动更新。

### 旧项目中可迁移的内容

| 内容 | 旧仓库路径 | 迁移方式 |
| --- | --- | --- |
| Panda | `assets/robots/panda.xml`、`assets/third_party/franka_emika_panda/` | 保留旧位置伺服、惯量、关节及物理限界 |
| 安装与 TCP | `configs/robots/panda.json` | 保留基座、复位关节、勺具挂载变换与 `tcp` site |
| Spoon | `assets/task/tableware/spoon/` | 完整带上视觉 mesh、勺头/勺柄碰撞 mesh；复用旧刚性装配 |
| Bowl | `assets/task/tableware/bowl/` | 复用旧固定、水平装配及碰撞参数 |
| Plate | `assets/task/tableware/plate/` | 保留备用，不创建默认场景 body 或接触 pair |
| Bean | `assets/task/foods/beans/meshes/bean_visual.obj`、`sim/beans.py`、`configs/scene.json` | 复用视觉 mesh + 同 body 原生 ellipsoid 碰撞体，数量严格为 1 |
| 接触 pair | `assets/task/tableware/contact_pairs.xml` | 只装配当前场景实际存在的勺/碗 geom；不带盘子引用 |
| 装配与恢复 | `src/feedingrobot/sim/model.py`、`sim/task.py` | 迁移必要装配、自然沉降和完整状态恢复能力，重建模型索引 |
| IK | `src/feedingrobot/control/adapter.py` | 复用 `FrameTask`、非臂 DOF 冻结、`solve_ik`、关节参考与位置伺服核心 |
| 归一化数值核心 | `src/feedingrobot/policies/dp_normalizer.py` | 搬用 `fit_linear_limits` 和 `linear_transform`，替换旧字段/旧数据集封装 |
| 任务与教师 | `sim/events.py`、`configs/task.json`、`experts/teacher.py`、`experts/bean_path.py` | 复用任务事实判据和路径几何，教师输出改接新 delta 控制契约 |

不迁移旧 DiT 网络、旧 checkpoint、旧策略输入 schema 或旧动作标签。旧 Panda 108 维观测含 Bean/嘴部/接收面的真实位姿、阶段和接触信息，不符合本次视觉策略输入要求。[S8][S9][S10][S11]

旧 `RobotAdapter.set_twist()` 的单位是 `m/s + rad/s`，并含速度、加速度、参考偏差和关节指令裁剪；旧 Gym 接口还检查动作是否落在 `[-1,1]`。因此只迁移 IK 核心，不能原样迁移整个 twist/Gym 动作链后声称满足本次 delta 和无后置限幅要求。

### 当前 conda 环境

本次只读核查确认已有 Python `3.12.14`、Torch `2.11.0+cu130`、Torchvision `0.26.0+cu130`、MuJoCo `3.14.0`、NumPy `2.2.6`、Diffusers `0.39.0`、Mink `1.3.0`、LeRobot `0.6.1`。当前没有 robosuite、h5py、zarr 或 hydra-core。

采用原生 MuJoCo 能直接保留旧物理与 IK；两个 camera 名称不要求安装 robosuite。实现时只补齐 HDF5 存储、配置及选定上游模块真正需要的依赖，不安装 DP/DP3 全套仿真 benchmark。沿用现有环境并生成匹配的依赖锁文件；当前 `environment.yml` 引用的 `requirements.lock.txt` 和可编辑安装所需项目定义尚不存在，需在环境阶段补齐。

当前会话 `torch.cuda.is_available()` 返回 `False`。训练阶段前重新核查 GPU/驱动可用性；CPU 可以做接口与短训练检查，正式训练预算由实际设备测定。

## 3. 物理场景与双相机

### 物理资产

保持旧 Panda、勺子、碗和单 Bean 的质量、摩擦、接触和尺寸参数。旧单 Bean 半轴为 `7 / 4.5 / 4 mm`，密度为 `1000 kg/m³`；不把视觉 mesh 自动当作新的碰撞模型。

复用旧装配器对勺子的刚性挂载和碗的固定处理，避免直接 include 原始带 `freejoint` 的资产而改变任务。plate 的文件和许可证完整保留，但默认 XML、接触 pair、reset 和任务逻辑均不引用它。

新模型重新核验单豆沉降、碗内稳定、勺头承托、真实交付和撤离。旧报告可以作对照，不能作为新增相机、delta 接口和新数据链已经通过的证明。

### 首轮初始状态随机化

固定布局仅用于接口调试，不用于 P7 正式验收。采用下列初始状态提议，范围是 FeedingRobot 的工程约定，不是 DP 论文统一规定：

- **碗位置**：以旧 `bowl_frame_position_m` 为基准，基座系 X、Y 分别提议 `U(-0.02,0.02) m` 偏移，Z 与姿态不变；每回合内保持固定。每次 reset 都从原始基准加本回合偏移，不能累计移动。
- **Bean 落点**：以旧 `reset_position_bowl_m` 的局部 XY 为基准，各轴独立提议 `U(-0.02,0.02) m` 偏移，旋转沿用旧 reset。出生高度遵循无穿透生成与自然沉降规则，不改变 Bean 物理参数。
- **有效性**：使用当前碗的真实有限碰撞几何，检查整颗 Bean 包络、出生净空、自然沉降后的碗内支撑与速度稳定；只因预定义的无效初态拒绝提议。碗底尺度限制意味着不能接受整个提议方形；接受分布应记录为“均匀提议经几何/沉降条件筛选”，不能仍称独立均匀分布。保存提议位置、拒绝原因和最终沉降位置，检查随机出生是否最终收敛到同一点。
- **一致性**：Bean 的世界出生位姿从当前 `bowl_frame` 转换，不继续读取旧固定碗原点。恢复状态时先恢复该回合碗的模型变换，再恢复物理状态；reset、教师几何、任务判定和双相机均使用同一实际布局。

P1 用中心、边界与角点检查桌面支撑、出生/沉降、reset 不累积和状态恢复；P2 用有效范围的极端样本检查两路图像和裁剪覆盖。借助独立校准初态验证教师覆盖，再在正式采集前冻结随机化规则以及开发/测试初态清单。不能因教师或策略失败缩小测试范围或换 seed。这里随机化的是场景初态，不增加图像数据增强。

### D435i 网格与 `<camera>` 分开建立

Menagerie D435i XML 只有机身网格、材质、质量和碰撞体，**没有 `<camera>` 元素或可直接使用的 RGB 光学内参**。[S12]

1. mesh/material 注册一次；建立两个不同名称的机身实例，避免重复 include 引发命名冲突。
2. 腕部机身挂到 Panda `link7` 的独立子 body，固定相对变换，随手部移动；俯视机身固定在世界坐标中的场景点。
3. 本次机身只用于视觉真实感：全部相机 geom 设置 `contype="0" conaffinity="0"`，移除机身碰撞体的物理作用；按视觉附件处理其质量，不引入源模型的 72 g 动力学负载。对照编译前后机械臂质量矩阵，确认未改变旧动力学。
4. 分别在 RGB 光学中心放置 `robot0_eye_in_hand` 和 `agentview`；光学中心根据 RGB 镜头网格核对后冻结，不把机身原点当作光学中心。
5. 采用固定外参，不通过 Bean 或嘴部真值驱动 camera `target/track` 模式。俯视光轴指向世界下方；腕部外参随手部刚性变换。

MuJoCo camera 的观察方向是局部 `-Z`，`+X` 向右、`+Y` 向上。安装方向、镜头位置、near/far clipping 和自身遮挡必须用渲染图核验。[S13]

### 视场、渲染与图像处理

按用户给定的 D435 RGB 近似水平视场 `69°` 配置透视相机。MuJoCo 的 `fovy` 是垂直视场，按输出宽高比换算：

```text
fovy = 2 * atan(tan(hfov / 2) * height / width)
```

本次 `256×256` 是方形成像，因此 `fovy≈69°`。这是选定的仿真方形视锥，不能描述为恢复了 D435 原生 16:9 图像；不先生成 16:9 再拉伸成方形。

每个 20 Hz 观测时刻，从同一 `MjData`、同一仿真时间渲染两幅 `256×256×3` RGB 图。渲染不调用物理推进，也不读取 depth。采集、训练、推理使用同一固定下采样方法：面积重采样到 `240×240`，除以 255 得到 `[0,1]` 浮点，裁剪到216后由固定图像 normalizer 映射为 CNN 使用的 `[-1,1]`。

训练仅随机裁剪 `240→216`；验证与推理采用确定性中心裁剪，每边12 px；四角极端裁剪位置为 `(0,0)/(0,24)/(24,0)/(24,24)`。不加颜色抖动、随机旋转、翻转、模糊、ImageNet 标准化或 RGB 的数据统计拟合。裁剪尺寸采用 DP Tool Hang 的240→216档位，`2x-1` 图像映射仍对齐 DP 官方；`256→240` 面积重采样是本任务的仿真适配。[S6][S16][S17]

策略图像显示真实视觉 mesh，隐藏碰撞调试几何、sites、坐标轴、camera frustum 和 viewer 叠加文字。旧 `overview` 可以保留作人工诊断，不能替代本次任一路策略相机。

视觉验收覆盖碗内、舀取、回平、携带、交付、撤离。保存两路 `256 / 240 / 216` 对照图，检查小 Bean 下采样后是否仍可辨识，裁剪是否切掉关键任务区域，以及手腕相机是否被机身/勺柄遮挡。视野不通过就调整并重新冻结安装参数，不能用物体坐标补足策略输入。

240档位变更记录：旧协议 `p2-calibration-v2` 的723个状态、7230条相机/裁剪测量显示最差联合面积1像素，真实舀起两路仅2像素，9/16/25候选全部失败。用户决定不锁K84/K76、不降低阈值，采用官方Tool Hang高分辨率档位；原图256、安装、69°视场、物理随机化及判据结构保持原值，旧定标包保留。新协议 `p2-calibration-v3-240` 在同一封存723状态上全量重测，原生240分割仅作验收仪表，216掩码与RGB使用同一裁剪。提供阶段×相机×240/216的全量分布、最小值和1%/5%/25%分位数、9/25/49及低分位RGB对照和最差1%帧，零帧不剔除。[S16][S17]

用户已依据v3全量定标锁定 `K240=8、K216=8`；原待定包保留，沿用v3协议且以不同输入指纹记录锁定与正式验收。未锁定的任何新协议仍保持待定。每个采样时刻至少一路同时满足两种尺寸门槛；五种裁剪独立检查，每阶段每路至少3帧达标。真实舀起、脱勺及前后1 ms必须满足同一联合判据；舀起前的可见帧不替代关键时刻验收。逐路保留遮挡区间×阶段×采样时长分布，部分遮挡与低于门槛分别记录；区间仅代表已渲染样本的端点跨度。阈值或安装变更均新建版本并全量重跑；正式验收通过后冻结安装，P4前十条新教师验证示教按同协议复验。实现与定标包契约见 [P2/P3说明](p2_p3_implementation.md)。

两步历史相隔50 ms，不能将 `T_o=2` 写成100 ms帧间隔。舀起前100 ms的实际信号量另以逐毫秒辅助回放验证，包含该事件前最近两个20 Hz观测网格时刻；不改变主723状态或扩大正式验收分母。

当前P2/P3已验收收口：K240/K216均8，安装冻结为 `p2-frozen-v2`，最终冻结配置的723状态、五种裁剪联合覆盖、30项关键事件和60组阶段证据全部通过；[正式结果与附件索引](p2_p3_implementation.md#k8正式验收与p2p3收口)。本轮只收口P2/P3；P4教师与数据采集留到下一轮，前十条新教师验证示教仍按相同冻结协议复验。

## 4. 观测、delta 动作与安全接口

### 最小策略观测

| 字段 | shape | 来源与处理 |
| --- | --- | --- |
| `agentview_image` | `[2,3,240,240]` | 固定俯视 RGB；接口 `[0,1]`，固定图像 normalizer 后 `[-1,1]`，编码前裁至216 |
| `robot0_eye_in_hand_image` | `[2,3,240,240]` | 腕部 RGB；接口 `[0,1]`，固定图像 normalizer 后 `[-1,1]`，编码前裁至216 |
| `robot0_eef_pos` | `[2,3]` | 旧 `tcp` site 的当前基座系位置，LinearNormalizer |
| `robot0_eef_quat` | `[2,4]` | 同一 TCP 的当前基座系姿态，LinearNormalizer |

首版本体状态为 `TCP xyz(3) + quaternion(4)`，共 7 维。旧 Panda 资产没有可动夹爪，故不虚构 `gripper_qpos` 或勺具状态。此处是对 DP 本体状态 schema 的 FeedingRobot 适配。

记录四元数顺序为 `wxyz`，采集与推理统一采用确定性的符号连续约定；不把轴角 delta 与四元数混用。若基座发生变换，先转换为基座系再形成状态与动作。当前旧场景基座与世界对齐，二者数值一致。

策略不读取 Bean/碗/接收目标位姿、接触、力矩、阶段、成功标签、任务 seed 或未来场景驱动。这些只供教师、物理事件、安全监控和离线诊断使用。两步历史是连续的 20 Hz 观测，相隔 `0.05 s`，不是两次重规划时刻相隔 `0.4 s` 的图像。

### 六维动作契约

```text
a_t = [Δx, Δy, Δz, Δr_x, Δr_y, Δr_z]
Δxyz: 米，基座系
Δr: 轴角旋转向量，弧度，基座系
```

对执行时刻的实测 TCP 位姿构造固定目标：

```text
p_target^B = p_actual^B + Δp^B
R_target^B = Exp([Δr^B]×) * R_actual^B
```

旋转左乘与旧基座系 twist 的旋转约定一致。必要时将目标变换回世界系送给 Mink `FrameTask`。每条 delta 只应用一次，随后 IK 在 50 个物理子步中追踪这个目标；不能每个物理子步重复加同一 delta。

不将 delta 当作 `m/s`、`rad/s`，也不隐含乘上旧 twist 的速度上限。无需新的勺具 actuator；舀取、携带和释放均由六维位姿完成。

### 三层执行边界

1. **策略层**：DDIM 完成采样 → 动作反归一化 → 截取 8 条物理 delta。此后不加 `clip/tanh`、速度/加速度范数限幅、参考误差限幅、平滑或旧 Gym 的 `[-1,1]` 拒绝规则。
2. **IK/物理层**：复用旧 IK、关节配置/速度可行性约束、非臂 DOF 冻结和位置伺服。保留模型原有 joint range、`ctrlrange/forcerange`；这些是机器人可行性与物理能力，不是策略动作后置归一化。不得静默照搬旧 Python `np.clip(reference.q, ...)`；求解或参考越界应记录失败，不用额外裁剪掩盖。
3. **安全层**：每个物理子步检查旧力保护、穿透、碰撞和数值故障。触发后停止余下动作、急停并终止回合，记录原因和触发前原动作；不生成缩小、绕行或恢复动作。旧运行时避让/参考修改不能混入新策略命令链。

教师采集、动作重放和策略推理使用完全相同的 delta、IK 和安全接口，避免训练数据执行一条链、推理执行另一条链。IK 无法完整到达请求目标时记录目标/实测误差，不能通过改写动作标签把误差隐藏掉。

## 5. LinearNormalizer 的准确复用

DP3 使用的就是 DP 系列 LinearNormalizer。迁移旧仓库 `dp_normalizer.py` 的两项数值核心，不迁移其中 108 维字段、阶段 identity、历史年龄或 train-window 数据封装。[S7][S10]

对动作 6 个通道和本体状态 7 个通道分别拟合，采用 FP32、`mode='limits'`、`last_n_dims=1`、`fit_offset=True`、`range_eps=1e-4`。

普通通道：

```text
scale_i = 2 / (max_i - min_i)
offset_i = -1 - scale_i * min_i
normalize(x_i) = x_i * scale_i + offset_i
unnormalize(y_i) = (y_i - offset_i) / scale_i
```

若 `max_i-min_i < 1e-4`，严格按官方处理：`scale_i=1`、`offset_i=-min_i`，只平移、不缩放；严格常量映射为 0。不能替换为除以 epsilon。

动作/本体状态归一化和反归一化均不 clamp。观测超出拟合区间时可以落在 `[-1,1]` 外；这不应触发动作裁剪。RGB 不参与上述数据 min/max 拟合，而是为两个图像字段分别使用 DP 的 `get_image_range_normalizer()`：固定 `scale=2`、`offset=-1`，将 `uint8/255` 后的 `[0,1]` 映射至 `[-1,1]`。[S6]

保存 `scale/offset/min/max/mean/std`、字段顺序、单位、数据源哈希。归一化参数不参与学习，checkpoint 与 EMA 模型共享同一份冻结参数；验证、恢复训练和推理不重新拟合。

P4 打印并保存 6 个动作通道和 7 个本体状态通道的诊断表：原单位 `min/max/range`、`scale/offset`、近常值标记（`range<1e-4`），以及归一化后的 `min/max`、1%/50%/99% 分位数和越出 `[-1,1]` 的比例。按训练/验证划分和任务阶段另列 1%–99% 活动区间，数据取未重复的原始行，测试源此时不读取。普通通道在完整拟合源上的 min/max 必映射到 `-1/+1`；“有效区间很窄”应指主体分位区间或某一阶段，不是这两个拟合端点。近常值/低活动通道先核对单位、字段与示教覆盖，不据此改换归一化方法或添加限幅。

统计范围也需明确：DP3 官方 dataset 在完整开发示教 replay buffer 上拟合，不应用内部 `train_mask`；旧项目的 `fit_dp_limits()` 则是 train-only。因此，本次迁移**数值核心**，拟合包装采用官方 DP3 的完整开发 buffer 语义，含该开发源中的内部 validation。独立 test 初始状态与测试示教在此源之外，绝不参与拟合。[S7]

若以后改为 train-only 拟合，必须记录为评估协议变更，不能再宣称统计范围也完全复刻 DP3。

## 6. CNN 与 Time-series Diffusion Transformer

### 视觉条件编码

沿用 DP 官方图像 Transformer 的视觉设计：

- 两个视角分别使用独立 ResNet-18，从头训练，不共享权重；P5输入为216×216，骨干结构保持原方案。Tool Hang高档位同样使用ResNet18，不据分辨率升级更换网络。[S17]
- BatchNorm 换为 GroupNorm，兼容 EMA。
- SpatialSoftmax 提取 32 个关键点，每视角得到 64 维特征。
- 同一时刻的两路视觉特征与 7 维归一化本体状态拼接：`64+64+7=135` 维。
- 两个历史时刻各形成一个条件 token；视觉只在每次重规划开始编码一次，十次 DDIM 更新复用条件特征。

DP 官方 image dataset 为 RGB 注册 `get_image_range_normalizer()`，policy 的 `normalizer.normalize(obs)` 显式执行 `2x-1`。本项目按此将 CNN 输入固定为 `[-1,1]`；如果只保留 `[0,1]`，应认定为有意偏离官方实现。本次不采用该偏离。映射仅执行一次，采集、验证和部署共用字段配置，避免在预处理和 policy 内重复映射。[S5][S6]

### Transformer 基线

| 参数 | 固定起始值 | 来源 |
| --- | --- | --- |
| `n_layer` | 8 | DP 图像 Transformer |
| `n_head` | 4 | DP 图像 Transformer |
| `n_emb` | 256 | DP 图像 Transformer |
| `p_drop_emb` | 0.0 | DP 图像 Transformer |
| `p_drop_attn` | 0.3 | DP 图像 Transformer |
| `n_cond_layers` | 0，条件使用 MLP | DP 图像 Transformer |
| `causal_attn` | True | DP 时间注意力 mask |
| `time_as_cond / obs_as_cond` | True / True | DP 条件 decoder |
| `pred_action_steps_only` | **False** | 必须预测完整 16 步 |
| 输入/输出动作 shape | `[B,16,6]` | 用户动作与窗口要求 |

每层对动作 token 做因果时序 self-attention，再对观测 memory 做 cross-attention；扩散 timestep 经 sinusoidal embedding 作为条件 token，动作序列具有位置编码。采用官方时间与条件 mask，不额外引入大模型、旧 adaLN-Zero DiT 或新增感知模块。[S5]

## 7. x₀ 训练与 DDIM 推理

### 网络直接回归干净动作

设物理动作窗口为 `A`，归一化后 `x₀=N_action(A)`。每个训练样本随机选择 `k∈{0,…,99}`、采样高斯噪声 `ε`，由 scheduler 的前向扩散得到：

```text
x_k = sqrt(ᾱ_k) * x₀ + sqrt(1-ᾱ_k) * ε
x̂₀ = Transformer(x_k, k, observation_condition)
loss = mean((x̂₀ - x₀)^2)
```

**loss 直接作用在网络输出 `x̂₀` 上**，不是 `ε̂` loss，也不先用 scheduler 更新结果计算监督误差。只对动作序列扩散，RGB 和本体状态作为条件，不做观测重建或额外辅助 loss。训练 loss 前不裁剪网络输出。[S4]

“训练 100 步”指训练扩散时间网格包含 100 个 timestep；每个 batch 对抽到的 timestep 进行一次去噪监督，并非每个 batch 串行运行 100 次 Transformer。

### DDIM 配置

```yaml
num_train_timesteps: 100
num_inference_steps: 10
beta_start: 0.0001
beta_end: 0.02
beta_schedule: squaredcos_cap_v2
prediction_type: sample
timestep_spacing: leading
steps_offset: 0
set_alpha_to_one: true
clip_sample: true
clip_sample_range: 1.0
thresholding: false
rescale_betas_zero_snr: false
# scheduler.step arguments
eta: 0.0
use_clipped_model_output: false
```

推理从 `[B,16,6]` 标准高斯噪声开始，时间索引固定为：

```text
[90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
```

每次网络给出 `x̂₀`，将它和当前 `x_k`、timestep 交给 `scheduler.step()`，由 scheduler 换算成一步 DDIM 更新。`sample` 分支应解析地计算方向项：

```text
ε̂_k = (x_k - sqrt(ᾱ_k) * x̂₀) / sqrt(1-ᾱ_k)
```

该 `ε̂` 是 scheduler 的内部换算，不是网络预测目标。`eta=0` 时，DDIM 在给定初始噪声和模型条件后不再添加逐步随机噪声。

保留 DP3 原有 `clip_sample=True`：scheduler 在每次更新内裁剪预测的干净样本至 `[-1,1]`。这是**采样器内部行为**，不等于反归一化后的动作后置限幅。loss 仍对未经此裁剪的 `x̂₀` 监督；最终执行链不再添加 clamp。[S4][S14]

DP/DP3 历史安装指定的 Diffusers `0.11.1` 与当前 `0.39.0` 的 `sample` 方向项实现存在差异，旧版也没有显式 `timestep_spacing` 参数。本项目保留现有环境，验证正确的 `x̂₀→DDIM` 换算并锁定版本，不能整套照搬旧环境或声称与旧版本采样结果逐位相同。[S14][S15]

### 滚动执行和索引

按官方完整窗口约定，重规划时窗口对齐为：

```text
观测条件：o_(t-1), o_t
16 条动作窗口：a_(t-1), a_t, ..., a_(t+14)
反归一化后取 action_pred[:, 1:9]
执行：a_t, ..., a_(t+7)
```

索引 0 对齐较早观测，不能误执行。其余未来 7 条在本轮不执行；8 步后使用最新两个连续观测重新预测。`horizon=16` 指完整预测窗口，不表示 16 条全部待执行未来动作。[S5][S4]

reset 后只有 `o₀`，首次历史按官方不足长度处理初始化为 `[o₀,o₀]`，双路 RGB 与本体状态同步重复首帧，首轮仍执行 `action_pred[:,1:9]`。随后每条执行动作后更新连续观测历史，检查 success、failure 和安全层；一旦终止，丢弃剩余 chunk。同步仿真中推理期间不额外推进物理；同时记录仿真时间与真实耗时，不能用“每 0.4 秒仿真时间重规划”替代实际吞吐或实时性测量。216相对76输入像素量约为8.08倍；P5/P6必须实测CNN编码、采样及完整重规划耗时的分位数，并记录设备与测量范围，不能据像素量估算默认GPU或实时性达标。

## 8. 示教、图像存储与时间对齐

### 示教采集

复用旧单豆教师的舀取、回平、携带、渐进释放和撤离几何路径，迁移到新 20 Hz delta 接口。教师可以使用当前仿真真值规划示范；部署策略不能访问这些字段。

动作标签必须是**当时下发的 delta 命令**。不能用事后实际 TCP 位移替代命令，也不能直接把旧 twist 标签改名为 delta。保留教师自身的路径、时间和速度规划，先生成**下一条 50 ms 控制步的参考位姿**，再计算它与当前实测 TCP 的基座系差值，以及 `Log(R_target * R_actual^-1)`；通过新接口执行并验收。不能将旧教师的整段 waypoint 直接相减当作单条 delta；教师内部轨迹规划约束不属于部署策略的后置限幅。

Bean 随机化不能只改 reset：旧 `pickup_path()` 的入勺和扫掠坐标基本固定。教师必须在自然沉降后，根据实际 Bean 落点和当前碗几何重新生成入勺、扫掠与沿壁抬升的连接路径，检查完整勺具净空，再沿用旧携带/交付/撤离方法。不能只平移整条固定路径或仅在成功后读取 Bean 真值。[S11]

首批先完成单条全流程重放，再在随机化初态下采集 10 条独立有效示教用于管线验证。正式开发集初始预算为 50 条不同初态的有效轨迹，按 episode 固定划分 40 条训练、10 条验证；数量与划分是本项目起始预算，不是文献保证成功的门槛。记录教师失败率与落点区域覆盖，不能复制轨迹凑数或无记录地重采样直到成功。校准若暴露有效初态的教师失败，应修正教师并复验；独立测试清单不按教师成功与否筛选。

所有进入训练的示教必须通过当前物理任务完整成功判据。失败轨迹另存用于诊断，首版不混入成功示教的模仿学习目标。

P4 对原单位的 `Δr=Log(R_target * R_actual^-1)` 保存范数直方图/经验分布、分位数、最大值和接近 `π` 的样本数（检查区间先定为 `‖Δr‖≥π-0.1 rad`），并检查转弯处逐通道时间曲线。对近 `π` 或相邻步跳变的样本回查 50 ms 参考目标、SO(3) 运算和四元数约定；修正标签生成后重新验收，不通过动作裁剪掩盖分支或 waypoint 错误。范数图是检查工具，不能单独证明没有分支跳变。

### P4 后立即运行固定动作回放对照

教师数据可重放、共享 delta/IK/安全链通过后，立即运行，不等到 P7 或正式训练。预先按开发示教 ID 顺序选取第一条成功示教，冻结其 ID、动作序列哈希、终止与超时规则；不根据回放成绩重新挑选轨迹。通过同一控制链顺序执行这条六维动作序列，不读取 RGB、不根据布局改写动作。

早期对照使用与正式数据同分布的 **50 个独立开发诊断初态**，预先冻结清单，不读取或占用 P7 测试初态，也不将诊断数据加入 normalizer 拟合源。保存固定回放的完整成功数/50，以及下述阶段指标。若完整回放成功率 `R_diag≥50%`，判定当前随机化范围或判据不足以排除固定轨迹；扩大范围、重新验证几何/教师/视野后，新建并冻结协议版本，重建开发/诊断/测试初态清单，按新范围重新采集示教并拟合 normalizer，再重跑此对照。保留旧版本数据与结果，不将旧数据无记录地混入新拟合源。该检查属于 P4 收尾验收，不阻塞 P0–P3 开工。

阶段结果同时给出全回合分母的 pickup、携带到既定待喂位、delivery、撤离通过率，以及各段“进入后成功数/进入数”的条件成功率；没有进入该段时记为 N/A，不能写成 0 或 100%。首轮接收面/嘴部位置固定、不做随机化，携带与交付段条件成功率高是预期现象；**舀取段**才是本次布局变化的主要区分段，不能用后段成绩单独证明视觉依赖。完整成功率的预设阈值仍按全回合分母计算。

### 首版直接存 RGB 帧

采用 robomimic 风格 HDF5 episode 结构，直接保存两路渲染帧，避免训练依赖重渲染是否确定：

```text
data/demo_<id>/
  obs/agentview_image            uint8 [L,256,256,3]
  obs/robot0_eye_in_hand_image    uint8 [L,256,256,3]
  obs/robot0_eef_pos             float32 [L,3]
  obs/robot0_eef_quat            float32 [L,4]
  actions                       float32 [L,6]   # 米 / 弧度
  timestamps                    float64 [L]    # pre-action simulation time
  states                        完整仿真快照/恢复记录
  dones / success / failure_reason
```

用无损压缩保存图像；任务真值和安全信息放独立诊断字段，不进入 policy 的读取白名单。metadata 保存模型/相机/控制/教师版本、reset、seed、碗位姿、Bean 提议与沉降位姿、字段和坐标约定。

每一步先记录同一时刻的双 RGB 与本体状态 `o_t`，再记录并执行 `a_t`，随后推进 `0.05 s`。旧教师动作是 20 Hz，而旧部分状态日志/Gym 为 50 Hz；本次统一在新 20 Hz 边界重采集，不能跨频率直接拼接。[S9][S11]

训练读取时固定面积重采样到240；不把随机裁剪结果永久写回数据文件。按完整开发集原始动作/状态行拟合 normalizer，不因滑窗重复或 padding 改变统计权重。

最坏 60 秒 episode 有 1200 个控制步，两路原始 RGB 约 `450 MiB/episode`；50 条约 `22 GiB`，未计压缩与状态。先用真实采集测定压缩率和容量，再开展正式采集。

### 状态重放重渲染作为后续优化

旧 `set_state` 能力可以迁移，但“只存状态、训练前离线重渲染”不是首版默认。若以后启用，完整快照须覆盖 qpos/qvel、actuator 状态、外力、仿真时间、场景驱动、随机状态以及相关控制器状态，并检验相同模型/版本/渲染参数下恢复后的双 RGB、接触和后续动力学一致。随机碗位姿等 `model.body_pos` 修改不包含在 MuJoCo integration state 中，必须先恢复对应模型变换再恢复物理状态；直接存帧的首版重放也必须遵循这一要求。只存 qpos/qvel 不足以声称确定性重放。

### 滑窗与划分

按 episode 划分，禁止同一 episode 的不同帧进入训练和验证两边。采用官方序列采样语义：`sequence_length=16`、`pad_before=T_o-1=1`、`pad_after=T_a-1=7`，边界重复端点，不跨 episode。训练仅从两步历史生成条件，不能读未来 RGB 或状态。

独立测试初始状态/示教放在开发源之外，使用与开发集同一冻结随机化规则、互不复用的初态 seed 与清单。为保持 DP3 官方统计范围，开发源内部验证数据参与动作/本体状态 normalizer 的范围拟合，但不参与梯度更新；这一点在结果报告中明确披露。RGB 映射是固定参数，不由任何划分拟合。

### 随机源分离与记录

| 随机源 | 用途 | 独立性与记录 |
| --- | --- | --- |
| 提议 RNG | 碗/Bean 初态及拒绝后的下一提议 | 按 `(协议版本, split, episode_id)` 派生独立子流，候选编号只推进该 episode 流；保存 seed、候选编号、拒绝记录与实际初态 |
| 教师 RNG | 教师中确实存在的随机选择 | 独立 seed/state；确定性教师记为未使用，不从提议 RNG 借随机数 |
| 采样器 RNG | 训练样本顺序、shuffle/采样 | 独立 generator；记录 seed、状态及恢复时的 sampler 位置 |
| 训练 RNG | 网络初始化、扩散 timestep/噪声、dropout、随机裁剪 | 分别记录训练进程及 worker 的 Python/NumPy/Torch CPU/CUDA 随机状态；worker seed 不借用采样器流 |
| 验证 RNG | 固定验证窗口的 timestep/噪声 | 独立 seed，按窗口 ID 固定；验证不消耗训练随机流 |
| DDIM 推理噪声 RNG | 每次预测窗口的初始高斯噪声 | 闭环验证、诊断、测试分开；按初态 ID/重规划序号固定 seed 规则，`eta=0` 也需要此源 |

各类使用独立 generator/命名空间和不同 seed，保存配置中的 seed 清单及相关 state，不共用一个全局随机流。不同训练 seed 使用同一份冻结数据与初态；额外验证、提前终止或拒绝采样不得改变后续训练样本顺序、噪声或其他 episode 初态。训练 checkpoint 保存全部在用随机源与 worker/sampler 的恢复信息，回放记录保存 episode 对应随机源；P0 固定分离规则，P4 检查记录可追溯。

## 9. 训练与评估协议

### 训练起始参数

网络是 DP Transformer，因此优化器和 EMA 采用 DP 图像 Transformer 起始配置，避免混用 DP3 U-Net 参数后却不说明来源。[S5]

| 参数 | 起始值 |
| --- | --- |
| optimizer | AdamW，按官方 Transformer 参数分组 |
| learning rate | `1e-4` |
| betas | `[0.9,0.95]` |
| Transformer weight decay | 可衰减参数 `1e-3`，按官方分组保留不衰减项 |
| CNN weight decay | `1e-6` |
| batch size | 64 |
| LR schedule | cosine，warmup 1000 optimizer steps |
| EMA | `inv_gamma=1.0`、`power=0.75`、`max_value=0.9999` |
| 首轮 seed | 42；完整稳定性评估用 3 个独立训练 seed |
| 正式初始预算 | 3050 epochs；实际 optimizer steps、样本量和耗时同时记录 |
| validation | 每 epoch 计算固定 timestep/噪声下的单次去噪 MSE：`val_denoise_x0_mse` |
| 闭环验证/保存 | 每 50 epochs，全部 10 条验证初态各 1 回合；保存 last 和按冻结规则选出的最佳 checkpoint |

先验证单 batch 的前向/反向、有限 loss 和 gradient，再在少量示教上检验能否学习动作序列，最后进入正式预算。若硬件容量导致 batch 或预算变化，写入配置与报告，不悄悄修改。

每 epoch 验证为便宜的**单次去噪 loss**：给每个验证窗口固定一个扩散 timestep 和高斯噪声，构造 `x_k`，在 `eval()` / `no_grad()` 和中心裁剪下前向一次，计算未经 scheduler 裁剪的 `MSE(x̂₀,x₀)`。验证使用独立固定随机源，不消耗训练 RNG；固定窗口/timestep/噪声保证跨 epoch 可比。此项不运行 10 步 DDIM，不做每 epoch 的完整采样动作 MSE；完整 DDIM 仅用于另行规定的闭环评估。[S5]

checkpoint 保存 CNN/Transformer、EMA、normalizer、optimizer、LR scheduler、epoch/global step、随机状态和配置/数据/资产哈希，保证恢复训练与独立推理使用同一方法。

### 完整任务成功与安全

沿用当前旧代码的单豆物理事件定义，并在新实现中重验：[S11]

- **pickup**：真实勺头支撑、Bean 完整高于碗沿、脱离碗接触，满足旧稳定速度条件，持续 `0.5 s`。
- **delivery**：Bean 完整进入接收区域、由接收面真实接触支撑、已经脱勺，持续 `0.2 s`。
- **success**：delivery 后勺具退出口部、解除接触，Bean 仍在接收区且脱勺，持续 `0.1 s`。
- 保留旧接触峰值 `5 N`、补偿腕力 `8 N`、穿透和掉落等失败条件；这些是 FeedingRobot 工程判据，不是 DP/DP3 超参数。
- 首版沿用 `60 s` episode 上限。阶段只做教师/任务判定，不驱动部署策略的动作或恢复流程。

不能用“豆子靠近目标”替代完整交付；也不能把旧 M4 数值事件比较容差 `1.1 s` 当成 delivery 确认时长。

### 选模与独立测试

用开发验证初始状态上的**完整闭环成功率**选 checkpoint，离线 MSE 只做训练诊断。测试初始状态不用于选模、调参或 normalizer 拟合。

训练开始前冻结选模规则：每 50 epochs 使用 EMA policy（`eval()`、中心裁剪），在全部 10 条验证初态上各跑 1 回合；每次恢复相同初态，并采用相同的按初态 ID/重规划序号确定的 DDIM 初始噪声规则。成功率粒度为 10 个百分点。checkpoint 按 **完整成功数降序 → pickup 成功数降序 → 更早 epoch** 排序；均以这 10 回合为分母，不追加回合打破并列，不再用离线 MSE 或测试结果改排序。

首轮完整评估建议 3 个训练 seed，各运行同一份预先冻结的 50 个有效随机测试初态，推理噪声规则和物理条件固定；这是 50 个不同初态上的 3 次训练重复，不能称为 150 个不同初态。测试只按事先规定的几何/沉降规则确定有效性，所有有效初态都进入分母，不因教师或策略失败换样本。报告每个 seed 和总计的成功率、pickup/delivery/retract 失败率、急停次数、IK 故障、平均完成时间、推理耗时分位数，以及碗/Bean 沉降位置的覆盖。

P7 将 P4 已冻结的同一条动作序列在这 50 个测试初态上再跑一次，与策略成对报告完整成功率、失败位置和相同的阶段指标。令 `R` 为固定回放的完整成功率，`P_i` 为第 i 个训练 seed 的策略完整成功率，`P_all` 为三次训练重复的合并成功率；以下阈值在训练前固定：

- `R≥50%`：判定随机化范围/判据不足；与 P4 的绝对阈值相同。
- 对已经达到任务门槛 `P_i≥90%` 的策略，若 `R≥P_i−10 个百分点`，即定义为“相近”，判定范围/判据不足；合并指标达到任务门槛时，也检查 `R≥P_all−10 个百分点`。任一合格 seed 或合并指标触发即记录，不用合并结果隐藏某个 seed 的情况。
- 若策略自身未达到 90%，先记录为策略/训练未达任务目标；不能仅因 `R≥P_i−10 个百分点` 就把低策略成功率解释为范围不足。绝对回放阈值仍独立有效。

触发范围/判据不足后保留原结果，扩大随机化范围并重新验证物理、教师、数据与视野，建立新协议版本，重新采集/训练/选模；使用新冻结、未参与调参的测试初态重新验收。已查看的旧测试清单归入诊断，不能继续标为新版本的独立测试。仅优于固定回放也不能证明 RGB 的因果作用；若报告作出视觉依赖结论，还需同一测试初态下的图像遮蔽或错配对照，并披露其观测分布变化。

本项目拟定工程目标为**未见随机初态**上的完整成功率 `≥90%`，并且没有被隐藏的动作修正；这是待通过实际实验验证的目标，不是已取得的结果或论文承诺。该指标证明冻结小范围分布内的任务表现，不单独证明使用视觉、更广布局泛化或新对象泛化。固定场景调试成绩必须单列，不能替代 P7；报告必须包含这些限定和回放对照结果。

## 10. 分阶段交付与验收

| 阶段 | 工作与产物 | 必须通过的检查 |
| --- | --- | --- |
| P0 环境/来源 | 项目定义、依赖锁、来源/许可证清单、配置来源矩阵、随机源清单 | 必需模块可导入；DDIM `sample` 分支及 `90…0` 时间索引检查；六类随机源独立且可记录 |
| P1 物理迁移 | Panda/Spoon/bowl/单 Bean 场景、随机 reset；plate 备用文件 | 场景可编译；只有 1 Bean；盘子未装配；范围中心/边界/角点有效性与沉降通过；reset不累积；布局恢复一致 |
| P2 双相机 | 两个视觉机身、两个 `<camera>`、256→240→216与固定图像normalizer | 腕部跟随、俯视固定；双帧同步；无depth/调试标记；不改变动力学；随机布局关键区域可见；CNN输入[-1,1]且仅映射一次 |
| P3 delta/安全 | 六维基座 delta → 旧 IK 核心 → 位置伺服 | 逐轴小增量验证坐标/单位/左乘；每条动作只应用一次；无后置clip；安全触发急停，不改写命令 |
| P4 教师/数据 | 新20 Hz教师、HDF5、normalizer、滑窗、早期固定回放 | 教师覆盖与重放；时间对齐/无真值输入泄漏；normalizer通道范围/分位区间；Δr检查；RNG记录；50诊断初态的完整回放成功率及阶段指标，执行R_diag≥50%规则 |
| P5 网络/采样 | 双216×216 CNN（ResNet-18不变）、Transformer、x₀ MSE、DDIM10、EMA | `[B,16,6]` 形状；attention/条件mask正确；loss针对干净动作；固定输入的一步DDIM更新与解析公式一致 |
| P6 闭环接通 | `[1:9]` 执行8步、每0.4s重规划、逐步终止检查 | 无 off-by-one；最近两帧相隔0.05s；不重复累加delta；采集/重放/推理共用控制与安全链 |
| P7 正式训练/测试 | checkpoint、单次去噪验证loss、冻结选模、随机测试与回放对照 | 10回合选模及并列规则；未见初态任务目标；测试分母不筛选；回放绝对/相对阈值及阶段解读；能力限定/版本一致 |

依赖顺序：`P0 → P1 → P2/P3 → P4 → P5 → P6 → P7`。P2 与 P3 可在共享冻结物理基线后并行。P4 验收前不扩大量采集；P6 接通前不开展长预算训练。

验证重点是接口和数值行为：normalizer 与上游对照、delta 单次执行、DDIM 解析更新、窗口时间对齐、输入白名单、同状态双相机和安全终止。避免为每个简单包装函数堆积重复测试。

## 11. 最小代码组织

保留旧项目 `feedingrobot` 包名，按已有职责迁移必要文件，不另造通用机器人框架：

```text
assets/robots/                         Panda
assets/task/                           Spoon / bowl / Bean / plate备用
assets/third_party/realsense_d435i/     相机网格与许可证
configs/                              物理、相机、动作、训练配置
src/feedingrobot/sim/                  装配、事件、RGB、状态恢复
src/feedingrobot/control/              delta适配、IK、安全终止
src/feedingrobot/data/                 HDF5采集、episode、序列采样
src/feedingrobot/policies/             normalizer、CNN、Transformer、sample policy
src/feedingrobot/scripts/              validate / collect / train / eval
tests/                                接口与上游数值对照
docs/                                 本规划、配置来源与使用说明
datasets/                             开发数据，与测试源分开
outputs/                              检查报告、训练、测试、失败视频
```

上游模块仅导入实际用到的算法部分并保留许可证。旧 `dp_normalizer` 的字段读取器、旧大型网络、历史审计框架和其他 benchmark 不成为新实现的隐含依赖。每项修改都应能对应本文的一条已确认要求或验收条件。

## 12. 依据链接

- **[S1] DP 文献**：[arXiv:2303.04137](https://arxiv.org/abs/2303.04137)，文题与 `refer.md` 一致；视觉条件、时序 Transformer 与滚动执行。
- **[S2] DP3 文献**：[arXiv:2403.03954v7 全文](https://arxiv.org/html/2403.03954v7)，3D 点云方法及附录参数。论文主实验 `4/2/3` 与当前官方代码 `16/2/8` 有区别；本项目窗口按用户要求固定。
- **[S3] DP 官方仓库**：[固定提交](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)。
- **[S4] DP3 配置与策略**：[dp3.yaml](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml)、[dp3.py](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py)。
- **[S5] DP 图像 Transformer**：[配置](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_transformer_hybrid_workspace.yaml)、[policy](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py)、[Transformer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/transformer_for_diffusion.py)。官方该配置默认 `H=10`、epsilon、DDPM100；本项目用用户要求与 DP3 设置覆盖，不能称原 YAML 原样复现。
- **[S6] 双视角、视觉编码与图像归一化**：[DP lift_image task](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/task/lift_image.yaml)、[robomimic 图像编码配置](https://github.com/ARISE-Initiative/robomimic/blob/v0.2.0/robomimic/scripts/generate_paper_configs.py)、[get_image_range_normalizer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/normalize_util.py#L23)、[image dataset 注册固定图像 normalizer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/robomimic_replay_image_dataset.py#L182)。
- **[S7] DP3 LinearNormalizer 与统计范围**：[normalizer.py](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/model/common/normalizer.py)、[dataset.get_normalizer](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/dataset/metaworld_dataset.py#L58)。
- **[S8] 旧 Panda 与物理装配**：[panda.xml](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/assets/robots/panda.xml)、[model.py](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/src/feedingrobot/sim/model.py)。
- **[S9] 旧控制与配置**：[adapter.py](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/src/feedingrobot/control/adapter.py)、[Panda 配置](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/configs/robots/panda.json)、[scene.json](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/configs/scene.json)、[task.json](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/configs/task.json)。
- **[S10] 旧归一化与状态接口**：[dp_normalizer.py](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/src/feedingrobot/policies/dp_normalizer.py)、[task.py](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/src/feedingrobot/sim/task.py)。
- **[S11] 旧教师与物理成功定义**：[teacher.py](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/src/feedingrobot/experts/teacher.py)、[舀取路径 bean_path.py](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/src/feedingrobot/experts/bean_path.py)、[events.py](https://github.com/Minashiki/Diffusion-based-Feeding-Robot/blob/56bd51a2c8966ab040fda60d9615cff4e9926725/src/feedingrobot/sim/events.py)。
- **[S12] D435i 机身模型**：[d435i.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/0059d4335f8156206f63a35662313385f7ad6d74/realsense_d435i/d435i.xml)、[README](https://github.com/google-deepmind/mujoco_menagerie/blob/0059d4335f8156206f63a35662313385f7ad6d74/realsense_d435i/README.md)。
- **[S13] MuJoCo 成像定义**：[camera XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)、[Python rendering](https://mujoco.readthedocs.io/en/stable/python.html#rendering)。实现时以已安装 MuJoCo 版本核验参数。
- **[S14] DDIM 数值实现**：[当前环境对应 v0.39.0 源码](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/schedulers/scheduling_ddim.py)、[API 文档](https://huggingface.co/docs/diffusers/api/schedulers/ddim)。
- **[S15] 历史依赖差异**：[DP3 INSTALL](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/INSTALL.md)、[Diffusers v0.11.1 DDIM](https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddim.py)。

- **[S16] DP Tool Hang 高分辨率任务与发布实验**：[固定任务配置](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/task/tool_hang_image.yaml#L6-L10)为双路240；[官方Transformer实验配置](https://diffusion-policy.cs.columbia.edu/data/experiments/image/tool_hang_ph/diffusion_policy_transformer/config.yaml)明确crop216、n_obs_steps2，发布文件字节SHA256为 `46689b3155419e9c7eb896fb62af02fb005c7f67a3998d741c7c5cde826dc9a4`。仅采用该成像档位，不迁入其其余训练超参数。
- **[S17] robomimic Tool Hang 240→216与ResNet18**：[v0.2.0固定配置代码](https://github.com/ARISE-Initiative/robomimic/blob/62ed2de905caeb9133136e4d14d810a8b6baa96c/robomimic/scripts/generate_paper_configs.py#L278-L286)区分简单任务84→76和Tool Hang240→216；[骨干设置](https://github.com/ARISE-Initiative/robomimic/blob/62ed2de905caeb9133136e4d14d810a8b6baa96c/robomimic/scripts/generate_paper_configs.py#L150-L157)采用ResNet18Conv。
