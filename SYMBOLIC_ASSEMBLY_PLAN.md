# 符号装配与视觉插接完整实施计划

## 1. 目标与边界

系统的数据流固定为：

```text
符号装配结构
    -> 仿真 GT visual grounding
    -> top-view directional goal mask
    -> ACT / Diffusion Policy
    -> 通用视觉对齐、下降与插接
```

reference brick 是已经装配并保持不动的参考积木；target brick 是 episode
中唯一需要抓起并安装的积木。根积木预先固定在底板上，不采集空板安装第一块
积木的步骤。

第一阶段只解决单步、固定预装配高度的插接：GT 预处理模块从底板外真实抓起
target，并以 Cartesian 路径将砖块中心搬运到目标 XY 正上方 60 mm；搬运保持
抓取后的砖块朝向，专家轨迹从该安全位开始，
只包含对齐、下降、接触和插接。Pick、长距离 transport、释放和归位均不属于专家
轨迹，也不进入未来训练数据；整个流程禁止瞬移 target。

暂不包含：真实视觉检测/分割/跟踪、深度输入、显式目标高度、双臂协同支撑、
大规模悬空结构，以及由学习策略完成板外抓取和长距离运输。

## 2. 已完成状态

### 2.1 结构生成

- 结构位于 `tasks/type1/<family>/<sequence>/`。
- family 包括 `basic`、`adjacent`、`multilevel`、`dense` 和 `bridge`。
- 当前每类 10 个，共 50 个验证规模样本。
- 每个样本目录只包含 `structure_start.json` 和 `structure_goal.json`。
- start 是 goal 的严格子集，二者只相差一个 target brick。
- 所有结构使用规范化局部网格坐标：整体包围盒的 `min(x,y,z)=(0,0,0)`；
  JSON 不编码结构在底板上的全局平移。
- 批量生成目录已加入 `.gitignore`，原始 `tasks/type1/example1` 保留。

### 2.2 当前验证结果

- 50/50 个 goal 通过 JSON 字段、BrickSim topology、底板边界、同层碰撞、
  逐层支撑和底板连通性检查。
- basic：单 reference、单 target。
- adjacent：reference 周围有 1～4 块已装配干扰积木。
- multilevel：目标层覆盖第 1、2、3 层，当前分布为 4/3/3。
- dense：目标附近沿随机 x/y 轴生成同层障碍及对应下层支撑，同时保留另一条
  平行夹爪通道和 1-stud 刚体插入间隙，迫使专家在部分任务中旋转夹爪 90°。
- bridge：target 同时连接两个 reference，覆盖 5 种支撑组合。
- 项目自动化测试当前为 55 passed。
- GT safe-start 专家已在 Isaac Sim 中通过代表性动态验证：`basic/1` 72 步、
  `multilevel/3` 71 步、`dense/1` 76 步、`bridge/1` 74 步，均由 BrickSim
  目标 connection（含 offset/yaw）确认成功。
- 完整的非记录阶段已用真实物理抓取验证：`basic/1` 完成 Pick/transport、72 步
  装配、release/retreat/home；`dense/1` 自动选择 `grasp_axis=y`（相对默认方向旋转
  90°），完成 Pick/transport、71 步装配、release/retreat/home。`multilevel/7`
  使用 1×8 长砖，按 64 mm 长边规划约 ±38 mm 的夹爪预张开量，完成 Pick/transport、
  72 步装配、release/retreat/home。释放和回零后连接
  仍保持有效。

上述部分动态结果来自旧的精确预对齐准备流程。准备流程现已改为只对准目标 XY、
保留抓取后的 yaw，因此需要重新进行全量 Isaac Sim 验证。静态结果只证明符号结构
和 BrickSim 拓扑合法，不等同于所有结构已经通过动态抓取、碰撞和插接验证。

连续初始 yaw 的专家验证接口已完成：demo 支持指定绝对角度或按 seed 随机角度，
批量验证器支持重复指定角度或为每个任务生成确定性随机样本，并将实际角度写入
HTML/JSON 报告和独立日志。启用 yaw 验证时，loose target 会先放到配置中 Storage
区域的中心，再绕世界竖直轴设置角度；这只规范化板外抓取起点，不改变底板、预置
结构、目标位姿、offset 或 connection yaw。代表性无窗口动态验证已通过：
`adjacent/1 @ 30°` 139 步、`dense/1 @ -135°` 106 步、`bridge/1 @ -45°`
和 `multilevel/1 @ 175°` 均完成物理抓取、保向搬运、精确对齐/插接、释放和归位。

连续 yaw 使用高空笛卡尔路标的已验证 IK 分支；抓取候选若在安全位到目标 yaw
之间存在腕部限位或 IK 分支断点，会在抓取前被拒绝。1×2 砖使用 2 mm 抓取高度和
1°/步、1.5°指令超前的专用高空旋转限制，避免小砖倾倒；其他尺寸仍使用 4°/步、
6°指令超前。当前阶段故意不把 1×2、2×2 等几何对称性作为等价成功条件，专家
仍严格执行结构 JSON/BrickSim connection 指定的目标 yaw；对称等价将在完成基线
数据验证后单独设计，不能静默折叠训练标签。

## 3. 固定接口约束

### 3.1 策略输入

```text
ACT / Diffusion Policy input
├── top-view RGB
├── 3-channel directional goal mask
├── selected wrist RGB
└── 10-frame robot-state history (10 x 23 = 230)
```

- 不输入深度。
- 不向 robot state 添加当前机械臂 ID。
- 策略不得读取结构 JSON、积木 ID、offset、yaw、连接接口或 GT 位姿。
- 只有仿真环境侧的 grounding 模块可以读取上述特权信息并生成 goal mask。
- selected wrist RGB 只取实际执行当前任务的机械臂相机。
- top-view RGB 和 goal mask 必须同分辨率、同裁剪、逐像素对齐。
- `Base_Plate.Position/Orientation` 是底板世界位姿的唯一来源；BrickSim 将规范化
  的局部结构居中放置，结构 JSON 不得改变底板世界位姿。

单帧 23 维状态沿用 `AssemblyObservation`：TCP position 3、TCP rotation-6D
6、TCP twist 6、gripper width 1、gripper closed 1、external wrench 6。历史按
从旧到新排列；episode 第一帧复制填充至 10 帧。状态属于所选机械臂，但不追加
arm index。

### 3.2 Directional goal mask

mask 使用 `uint8 HxWx3`，背景均为 0：

- R：目标积木顶面 footprint，区域内为 255。
- G：区域内沿目标局部 +x 单调从 0 增至 255 的归一化坐标。
- B：区域内沿目标局部 +y 单调从 0 增至 255 的归一化坐标。

R 区分背景和局部坐标零值；G/B 同时编码朝向，避免单一轮廓对正方形或近对称
积木产生 yaw 歧义。mask 表示期望目标位姿，不做可见性遮挡裁剪，也不使用深度。

### 3.3 动作与数据阶段

- 第一阶段学习动作沿用局部插接模块的 6D TCP 增量：平移 3 + 旋转 3。
- 专家完成 Pick 和 transport，target 中心到达目标 XY 正上方 60 mm 后才开始记录；
  前置搬运不改变抓取后的 yaw。
- 记录 ALIGN、APPROACH、FIRST_CONTACT、PRESS、HOLD 到 COMPLETE。
- 抓取失败、掉砖、IK 失败、超力、超时或连接未建立的 attempt 全部丢弃。

## 4. 分阶段实施

### 阶段 A：GT grounding 与 mask（下一步）

1. 增加真正垂直于底板的 `Top_Camera`。现有 `Global_Camera` 是斜视审查相机，
   继续保留，但不能冒充 top-view policy 输入。
2. 从 start/goal 差分取得唯一 target ID；从 BrickSim topology 构建 target-centric
   单步连接计划。
3. 对 primary reference 调用 BrickSim `compute_connection_transform`，使用仿真中
   reference 的真实 world transform 得到 target 的期望 world transform。
4. bridge 等多支撑任务分别由所有连接计算目标位姿；各结果必须在位置和旋转
   容差内一致，否则拒绝该任务。primary connection 只作为计算入口，不忽略其他
   连接。
5. 从 Top_Camera 的实时内参和 world pose 计算 world-to-camera 投影，将 target
   顶面四角投影为像素多边形，并在多边形内栅格化 R/G/B。
6. grounding 结果通过独立的数据对象返回：目标 world transform、mask、target
   像素多边形和诊断信息；传给策略的观测只保留 mask。
7. 新增可视化检查命令，输出 top RGB、mask、叠加图，并在 Isaac 场景中显示半透明
   目标预览。命令支持一个 task 以及按 family 批量检查。

阶段 A 验收条件：

- 所有 50 个任务均能产生有限、非空、尺寸正确的 mask。
- R 区域在画面内，面积与对应砖块尺寸和投影一致。
- yaw 改变时 G/B 方向随目标局部坐标轴同步旋转。
- 多层目标的投影使用真实目标高度，而不是固定底板平面。
- bridge 的多连接目标位姿一致。
- 叠加图中的 mask 与场景目标预览逐像素对齐；允许的边界误差不超过 2 px。
- mask 生成不调用检测、分割、跟踪或图像配准网络。

### 阶段 B：固定高度专家单步插接（核心实现已完成）

1. 场景加载时 target 仍位于底板外；在专家轨迹开始前，GT 预处理器选择完整
   Pick/transport IK 链可达的机械臂，真实闭爪、Cartesian 抬升并搬运到目标上方
   60 mm，过程中不瞬移 target，不执行 yaw 对齐。
2. 每帧根据 target brick 的 GT 位姿重新捕获实际 `brick->TCP` 变换，闭环计算
   `T_goal_tcp = T_goal_brick * T_current_brick_to_tcp`。
3. 从安全位先保持约 60 mm 高度完成 XY/yaw 对齐，再单调下降、接触锁定、有限
   压装和连接验证；180°旋转使用高位目标 IK 分支消除旋转方向歧义。首次对齐采用
   严格阈值，下降阶段使用滞回并同步修正小误差，避免 ALIGN/APPROACH 抖动；错误
   offset/yaw 立即失败，不等待 episode 超时。接触后的失配必须连续 3 帧确认，随后
   完整退回约 10 mm 的无接触高度重新对齐，禁止用 0.5 mm 往返试探生成抖动轨迹。
   距离装配面至少 20 mm 时，高空对齐可使用 4°/步和 6°指令超前；易倾倒的 1×2
   砖使用 1°/步和 1.5°指令超前；低空修正、下降与接触阶段继续使用 2°限制，不能
   通过整体提高插接速度来缩短 episode。
4. 对每类先运行一个代表任务，再运行全部 50 个任务；记录失败阶段、IK 位置/
   旋转误差、接触力、是否掉砖和最终 BrickSim connection。
5. Pick/transport 属于数据记录前的 episode setup，不得混入训练轨迹；连接成功
   后的开爪、垂直撤离和回零属于数据记录后的 cleanup。三段日志和计步相互独立。
6. 抬升稳定后读取实际 `brick->TCP`，随后所有搬运段始终相对同一基准检查，禁止
   分段重置基准来掩盖累计滑移。抬升、搬运和撤离使用逐帧闭环 Cartesian 路径，
   避免关节插值的 TCP 弧线把砖拖出夹爪。
7. `grasp_axis` 表示真实手指开合方向，并映射到 Piper 工具局部 y 轴。每指预张开
   量按“目标夹持宽度的一半 + 6 mm 单侧余量”计算，而不是统一全开。高长宽比砖
   优先跨长边夹持以增加抗偏航力臂；紧凑砖优先短边，目标侧障碍约束优先级最高。
8. 抓取 TCP 位于砖底上方 2 mm 的侧壁区域。闭爪必须验证接触宽度；首次抬升允许
   有限竖直接触就位，随后搬运分别监控夹紧轴、手指轴、竖直和旋转滑移。夹紧轴
   夹紧轴允许不超过半个夹持宽度、上限 8 mm 的单侧接触就位，并使用 0.2 mm
   仿真比较容差；手指轴、竖直、旋转以及最终 5 mm safe-start 位置验收保持
   不变，因此放宽中途检测不会接受明显掉砖或错误装配起点。
9. 搬运拆分为安全高度提升、保持抓取姿态的长距离平移和下降到目标 XY 正上方
   60 mm；不执行 yaw 对齐，也不人为添加 XY/yaw 扰动。无负载关节步长为 0.02 rad，
   持砖 Cartesian 步长为 2 mm，指令超前量在 15 帧内渐增到 4 mm。
10. 所有创建 `Env` 的 BrickSim 命令都在 `finally` 中暂停仿真并调用不可取消的
    Kit quit；成功和异常路径均不得遗留 Isaac Sim 后台进程。

阶段 B 验收条件：固定初始条件下 50/50 各完成一次；加入计划内的初始 XY/yaw
扰动后，每类至少 100 次试验的成功率达到 95%，且无超力和未记录的失败。

### 阶段 C：LeRobot 数据采集

1. 每个 episode 加载一个任务，只生成预置结构和当前 target；后续积木不存在于
   场景中。
2. 在 transport 完成、target 位于目标上方 60 mm 且抓取稳定时开启记录。
3. 每帧保存 top-view RGB、directional goal mask、selected wrist RGB、230 维
   状态历史和 6D 动作。
4. teacher phase、结构 family、任务路径、执行臂、GT 位姿和失败原因仅写入训练/
   审查 metadata，不进入 policy observation。
5. 数据划分以结构和连接关系为单位，而不是随机拆帧；同一 episode 不能跨 split。
   split 清单存放在任务目录之外，保持每个 Type-1 样本只有两个 JSON。
6. 先采集小规模 smoke dataset，检查相机、维度、时间同步和 episode 边界，再扩大。

阶段 C 验收条件：数据集中只存在完整成功 episode；所有图像 shape/dtype 固定；
mask 与 top RGB 同步；state 恒为 230 维；无跨 episode 历史污染；训练/验证/测试
之间不存在结构或关系泄漏。

### 阶段 D：ACT / Diffusion Policy

1. 先用同一数据划分训练 ACT 基线，再训练 Diffusion Policy；两者共享完全相同的
   输入预处理、状态归一化、动作定义和评估协议。
2. 图像编码器分别处理 top RGB、goal mask 和 wrist RGB，融合后与 230 维状态
   特征共同条件化动作序列预测。
3. 训练时不得加载 metadata 中的 GT 字段作为模型输入。
4. 保存模型配置、normalizer、相机名称、图像尺寸、历史长度和动作约束，确保推理
   契约可复现。
5. 首先在已见关系上验证闭环插接，再进行未见结构和未见连接关系测试。

### 阶段 E：泛化与增量复杂度

按以下顺序增加难度，每次只改变一个变量并保留前一阶段回归集：

1. 更广的板外 target 初始 XY/yaw，以及进入学习阶段时的预对齐误差。
2. 未见积木尺寸组合、offset 和 yaw。
3. 未见局部邻接和密集结构。
4. 更丰富的目标层高；仍不够时才加入显式高度或深度。
5. 悬空/部分支撑结构。
6. 需要第二机械臂稳定结构的双臂任务。

## 5. 验证矩阵与停止条件

每一阶段必须通过后才能进入下一阶段：

| Gate | 必须验证 | 失败时处理 |
|---|---|---|
| 结构 | JSON、拓扑、支撑、单 target、family 语义 | 修生成器并重新生成全量结构 |
| Grounding | GT 位姿、多连接一致性、投影、mask 方向 | 不启动专家和数据采集 |
| 专家 | 可达、抓取、固定高度插接、最终 connection | 不保存失败 episode |
| 数据 | 输入维度、同步、阶段裁剪、split 隔离 | 删除 smoke dataset 后重采 |
| 模型 | 闭环成功率、未见关系泛化、安全约束 | 单变量消融后再扩数据复杂度 |

任何阶段都不得用“运行时改变 target 初始位置”“向策略泄漏 JSON/GT”或“保留失败
轨迹作为成功示范”来绕过失败。

## 6. 当前可用命令

重新生成验证规模结构：

```bash
uv run python ./run/generate_symbolic_tasks.py \
  --output tasks/type1 \
  --seed 7 \
  --count-per-family 10
```

验证一个 goal：

```bash
uv run python ./run/validate_goal.py \
  tasks/type1/basic/1/structure_goal.json
```

查看一个结构并运行当前专家：

```bash
uv run bricksim ./run/demo_symbolic_assembly.py \
  --task-dir tasks/type1/basic/1 \
  --inspect-seconds 30
```

指定或随机化 loose target 的绝对初始 yaw（两者互斥）：

```bash
uv run bricksim ./run/demo_symbolic_assembly.py \
  --task-dir tasks/type1/dense/1 \
  --initial-yaw-deg -135 \
  --inspect-seconds 5

uv run bricksim ./run/demo_symbolic_assembly.py \
  --task-dir tasks/type1/dense/1 \
  --random-initial-yaw \
  --yaw-seed 7
```

只验证抓取和搬运到目标正上方，不运行后续对齐/插接：

```bash
uv run bricksim ./run/demo_symbolic_assembly.py \
  --task-dir tasks/type1/multilevel/7 \
  --prepare-only \
  --final-hold-seconds 10
```

直接从 60 mm 安全位运行唯一 GT 专家：

```bash
uv run bricksim ./run/demo_gt_assembly.py \
  --task-dir tasks/type1/basic/1 \
  --safe-height-mm 60 \
  --final-hold-seconds 2
```

无窗口批量验证连续 yaw，并在浏览器查看实时 HTML 报告：

```bash
uv run python ./run/validate_expert_tasks.py \
  --tasks-root tasks/type1 \
  --yaw-samples 3 \
  --yaw-seed 7 \
  --output validation_reports/yaw_seed7

xdg-open validation_reports/yaw_seed7/report.html
```

也可重复传入精确角度，例如
`--initial-yaw-deg -135 --initial-yaw-deg 30`；中断后用同一输出目录加
`--resume`，已通过的 task/yaw 组合不会重跑。

运行自动化测试：

```bash
uv run pytest
```

## 7. 下一项交付

先运行一次完整的连续 yaw 动态基线：50 个任务、每个任务 3 个确定性随机角度，
检查 HTML 中的失败是否集中在某一尺寸、机械臂或角度区间。该批量运行是验收，
不再改变 target yaw、结构或成功标准。完成后，下一项只实现阶段 A，不同时修改
专家策略或启动数据采集。阶段 A 的交付内容为：

1. 垂直 Top_Camera 配置与健康检查。
2. 可单元测试的 GT target pose 与 directional mask 模块。
3. 将 `Top_Camera_rgb` 和 goal mask 暴露给环境观测，但隔离所有结构特权字段。
4. 一个可运行的 grounding 可视化命令及 50 个任务的批量检查模式。
5. 对位姿、多支撑一致性、相机投影、mask 通道和观测泄漏的自动化测试。

完成上述验收后，将 goal mask 接入已验证的 safe-start 专家并批量运行全部 50 个
任务；在此之前不恢复数据采集或 ACT/DP 训练实现。
