# 依赖与物理事件驱动的多臂执行平台：重构设计

状态：已授权代码实施与离线测试，第一批离线基础正在落地；不是全部已实现能力声明。未授权仿真、ROS 运行时或实际运动。当前实施范围与缺口见 [第一批实施状态](event-driven-foundation-status.md)。

## 1. 目标与范围

上层离散规划器决定哪个机器人执行什么技能、操作对象、任务目标、参与者及真实依赖。本平台负责阶段展开、连续运动、并发协调、物理验收和失败反馈。输入格式尚未定稿，可重新设计，不承担旧 `execution_layers` 屏障语义兼容义务。

核心转换：

> 按层等待、技能各自发运动 → 依赖与物理事件驱动的异步阶段调度 + 唯一多臂运动协调与执行所有权。

必须实现：

- 无真实依赖的操作不因层号等待；前驱物理完成、资源及物理条件满足后，后继可以申请推进。
- 不同技能、不同独立任务可并行，至少覆盖两个 PICK、PICK 与 PLACE_DOWN。
- 非工作臂不默认保持关节；允许有授权的避让，但不得破坏支撑、接触或抓持义务。
- 新请求与在途轨迹协调，不把正在运动的另一臂当静态障碍。
- PICK 的选择考虑具体 PLACE_UP / PLACE_DOWN 后继，并在抓后以实测抓持复核。
- 区分目标、搜索、后置验收、执行和物理完成失败。
- 计划成功不能代替物理成功；取消请求不能代替停止证明。

本轮不处理 HANDOVER 的新架构适配；不得删除已有实现，也不得允许未适配实现绕过新执行所有权机制。新模式下遇到 HANDOVER 应在运动前明确拒绝；若保留旧运行模式，两种模式须会话级互斥。

本设计不授权重新分配机器人、改变任务目标或自动插入重抓、放回等新任务。需要这些操作时，应请求上层重新规划，或由另行审核的显式恢复策略授权。

## 2. 现状与重构依据

以下为静态代码及历史证据审查结论，不表示已运行验证。

| 当前实现 | 现状与缺口 |
| --- | --- |
| `src/rocobrick/piper_run.py` | 输入门禁、会话构造、技能装配和运行报告集中于入口；应拆为应用装配与生命周期 |
| `src/rocobrick/robot_operation_plan.py` | 验证依赖、层序与 schedule；新契约应分离真实依赖与调度建议 |
| `src/rocobrick/multi_robot_scheduler.py` | 按 `execution_layers` 等待；层内并发不等于依赖驱动推进 |
| `src/rocobrick/skills/pick.py`、`src/rocobrick/skills/place_down.py` | 有阶段与物理验收，但 Handler 自己驱动运动，跨技能阶段意图不可统一协调 |
| `src/rocobrick/unified_runtime_facade.py` | 已有共享场景、plan-only、自由空间与前后继过渡，但职责覆盖过宽 |
| `src/rocobrick/coordinated_ros_runtime.py` | 已支持联合目标；`maintain_pose` 会形成目标及全路径关节保持约束 |
| `src/rocobrick/execution/ros_session_worker.py` | 以 `coordinated execution requires an idle safe group` 拒绝执行重叠；不能直接删除检查 |
| `src/rocobrick/coordinated_execution.py` | prepare/start/cancel 和同步执行证据值得保留 |
| `src/rocobrick/execution/operation_state_ledger.py` | 物理状态账本值得保留，但须与任务租约、控制通道所有权区分 |

现有前后继过渡已保护共享场景、holding/supporting 连续性、物理到达与租约采用。新系统应推广这些保护，不绕过它们。

历史故障：

- `runs/place-up-headless-baseline-20261017T101644Z-c65614/` 中失败点是 PICK 的 precontact boundary。
- `runs/pick-repeat-headless-20261018T154044Z-967c0f/` 中 PICK 候选、边界和 lift 已成功，后续 SUPPORT pre-phase 规划失败。
- 上述 OMPL 日志包含 `Unable to sample any valid states for goal tree` 与 `Insufficient states in sampleable goal region`，结果汇总为 `CombinedMoveItPlanningError: combined MoveGroup plan failed: status=ABORTED error_code=99999`；没有发送运动。日志不能独自证明几何必然无解，应进一步诊断 IK、目标碰撞、联合约束和采样覆盖。
- `runs/pick-entry-frame-headless-20261018T152937Z-62724c/` 首个候选被后置净空验收拒绝：`GRASP_APPROACH_WORLD_CLEARANCE: piper_0_gripper_link1 vs /World/feeding_platforms/piper_0 at sample 0: 0.003874 m < 0.004000 m`；第二个候选成功。这与路径搜索失败、执行失败不同。

## 3. 上下层计划契约

### 3.1 语义

新输入建议命名为 `robot-task-plan/v2`，正式名称和字段在契约实施时冻结。本文 JSON 为设计示例，不是已存在的解析格式。

- 操作身份、机器人分配、实体身份、目标和真实依赖来自上层。
- `dependencies` 表达硬性先后条件；数组顺序不表达依赖。
- 不要求 `execution_layers`、`planned_schedule`。展示布局不得进入执行语义。
- 优先级、期限和同步要求独立于依赖表达。
- 对同一机器人或同一物体的并发请求，平台负责排他；若顺序具有任务意义，上层必须显式声明。
- 无依赖不等于必然并发；物理条件、资源与时空冲突仍可能导致等待。
- 计划、目标结构、标定和模型来源保持不可变，记录原始字节及摘要，派生阶段图单独记录。

```json
{
  "schema": "robot-task-plan/v2",
  "plan_id": "assembly-example",
  "operations": [
    {"id": "pick_a", "skill": "PICK", "executor": "piper_0", "participants": ["piper_0"], "object_id": "brick_a"},
    {"id": "place_a", "skill": "PLACE_DOWN", "executor": "piper_0", "participants": ["piper_0"], "object_id": "brick_a", "target_ref": "slot_a"},
    {"id": "pick_b", "skill": "PICK", "executor": "piper_1", "participants": ["piper_1"], "object_id": "brick_b"}
  ],
  "dependencies": [
    {"before": "pick_a", "after": "place_a", "condition": "physical_success"}
  ]
}
```

PICK_A 确认完成后，PLACE_DOWN_A 可以申请推进，而不等待 PICK_B。若 PLACE_DOWN_A 与在途 PICK_B 冲突，交给协调器错峰、改路径或联合重规划。

若确实要求两个 PICK 都完成后才能放置，就让每个放置操作显式依赖两个 PICK。不得依靠层号隐含表达。

### 3.2 验证与反馈

输入验证至少包括：身份唯一、依赖无环、实体与目标可解析、机器人与参与者一致、技能受支持、后继对象关系一致、约束无直接矛盾。无法仅靠静态信息确认的物理条件留到运行期，不伪造事实。

区分外部操作级依赖与内部阶段级条件：前者默认必须等到完整物理成功，不得因内部某阶段完成就提前释放。若将来支持外部依赖某个中间物理事件，必须成为显式契约。

不可行反馈包含原 operation ID、阶段、失败分类、已验证事实、当前物体持有者、停止状态、保留租约和上层可采取的重新规划范围。不得把搜索预算耗尽表述为已证明无解。

## 4. 目录结构与职责

```text
src/rocobrick/
├── __init__.py
├── app/                         # CLI、组件装配、运行生命周期
│   ├── cli.py
│   ├── run.py
│   ├── bootstrap.py
│   └── lifecycle.py
├── task_plan/                   # 上层输入与任务依赖
│   ├── schema.py
│   ├── parser.py
│   ├── validation.py
│   ├── dependency_graph.py
│   └── structural_binding.py
├── contracts/                   # 小而稳定的跨模块 DTO、事件和失败类型
│   ├── identity.py
│   ├── phases.py
│   ├── motion.py
│   ├── events.py
│   ├── evidence.py
│   └── failures.py
├── skills/                      # 技能阶段知识；禁止发送运动
│   ├── registry.py
│   ├── pick/
│   │   ├── program.py
│   │   ├── candidates.py
│   │   ├── geometry.py
│   │   └── verification.py
│   ├── place_down/
│   ├── place_up/
│   ├── support_top/
│   ├── wait/
│   └── common/
│       ├── target_context.py
│       ├── placement_geometry.py
│       └── observations.py
├── scheduling/                  # 哪些阶段有资格申请推进
│   ├── phase_graph.py
│   ├── scheduler.py
│   ├── readiness.py
│   ├── resource_manager.py
│   └── recovery.py
├── coordination/                # 多个运动意图如何共存
│   ├── coordinator.py
│   ├── permissions.py
│   ├── reservations.py
│   ├── conflicts.py
│   ├── arbitration.py
│   └── replanning.py
├── planning/                    # 给定问题的求解与独立验收
│   ├── problem.py
│   ├── backend.py
│   ├── goal_feasibility.py
│   ├── successor_feasibility.py
│   ├── time_parameterization.py
│   └── validation/
│       ├── collision.py
│       ├── constraints.py
│       ├── continuity.py
│       └── temporal.py
├── execution/                   # 唯一运动发送权
│   ├── manager.py
│   ├── ownership.py
│   ├── preparation.py
│   ├── trajectory_execution.py
│   ├── gripper_execution.py
│   ├── monitoring.py
│   └── cancellation.py
├── world/                       # 实测世界与物理事实
│   ├── snapshots.py
│   ├── versions.py
│   ├── physical_state.py
│   ├── attachments.py
│   └── scene_updates.py
├── model/                       # 机器人、工具、标定和来源
│   ├── robot.py
│   ├── tcp.py
│   ├── calibration.py
│   └── provenance.py
├── adapters/                    # 框架相关实现，禁止业务决策
│   ├── moveit/
│   ├── ros2/
│   └── isaac/
└── audit/                       # 不可变输入、事件与报告
    ├── input_capture.py
    ├── event_log.py
    └── run_report.py
```

这是职责布局，不要求一次创建所有空文件。模块有独立职责和测试对象才拆分。避免新建泛化 `utils.py`、万能 facade，或继续以 `unified_*`、`coordinated_*`、`dual_arm_*` 在根目录平行表达同一职责。

### 4.1 唯一职责归属

| 模块 | 拥有 | 禁止 |
| --- | --- | --- |
| `task_plan/` | 任务身份、分配与依赖校验 | 擅自改分配或生成轨迹 |
| `skills/` | 阶段程序、几何规则、候选域、证据验证器 | 发运动、决定其他技能排队 |
| `scheduling/` | 阶段状态、资格判断、任务资源租约 | 求解路径、控制机械臂 |
| `coordination/` | 运动准入、预约、仲裁和重规划事务 | 改任务目标、伪造物理完成 |
| `planning/` | 求解、候选评估和轨迹验收 | 发控制命令、释放任务资源 |
| `execution/` | 控制通道所有权与执行生命周期 | 把控制器成功当抓取或装配成功 |
| `world/` | 实测状态和物理事实有效性 | 根据规划成功创建 holding/connection |
| `adapters/` | 外部框架协议实现 | 自行调度、任务级重试 |
| `app/` | 装配和生命周期 | 堆积技能和求解逻辑 |

资源管理分三类，禁止混成重复的 busy 标志：

1. `scheduling/resource_manager.py`：任务租约、物体持有和支撑义务对应的资源占用。
2. `coordination/reservations.py`：时空预约、轨迹 revision 和冲突集合。
3. `execution/ownership.py`：控制通道唯一写入者、epoch 和 action handle。

物体实际由谁持有是 `world/physical_state.py` 的物理事实；谁有权推进相关任务是资源租约。两者需校验一致，但不是同一个字段。

### 4.2 依赖方向与扩展规则

- `contracts/` 只含共享数据、校验及小型协议，不导入业务实现或 ROS/MoveIt。
- 领域模块通过 DTO、事件和消费者定义的 Protocol 协作；不直接导入具体 adapter。
- `planning/` 不依赖具体技能 Handler；技能输出描述性候选域，通用后继评估消费该数据。
- `world/` 接纳经验证的证据，不反向导入技能；证据验证器由应用装配。
- `adapters/` 实现核心定义的端口，`app/bootstrap.py` 负责注入；禁止核心模块主动创建 ROS 客户端。
- 只有执行管理器的适配链有发送运动的权限。MoveIt 规划调用必须 plan-only，不能用隐式 plan-and-execute 绕过它。
- Python 3.11；优先扩展现有 dataclass、Protocol 和 schema；避免 `Any`、宽泛 cast 和大规模兼容转发。

## 5. 核心契约

正式字段类型与序列化 schema 在阶段 1 冻结。以下是必须覆盖的语义，不是已存在 API。

### 5.1 PhaseIntent

- 标识：plan / operation / phase / attempt ID、输入摘要。
- 依赖：内部前驱、所需物理事实及其有效性。
- 目标：位姿或关节区域、相对关系、接触目标、候选集合或连续候选域。
- 约束：全路径要求、工具和携带物包络、接近与退出通道。
- 接触：明确物体/link 对、接触模式、适用阶段、进入和退出条件。
- 权限：每个相关机械臂可修改的自由度和不可破坏的义务。
- 资源：阶段所需租约及前驱移交关系。
- 完成：控制器终态要求与独立物理证据谓词。
- 中断：安全停点、不可中断区间、撤销和恢复规则。

技能可以根据新观测生成后续意图；不应在任务开始时冻结所有具体位姿。候选生成不等于运动授权。

### 5.2 MotionProblem

由协调器组合一个或多个阶段意图：相关机械臂、允许自由度、统一快照、候选与硬约束、已有轨迹承诺、开始时间窗、冻结前缀、可修改后缀和求解预算。不能包含 MoveIt 消息类型。

### 5.3 MotionCommit

经过规划、验收、租约与版本复核后产生：轨迹及 revision、参与机器人、开始时间与切换点、场景依赖、权限、预约、租约绑定、执行 epoch、停止处理和验收摘要。

执行器只接受有效承诺。同一 commit 重复投递不得重复运动；旧 epoch、旧 revision、已撤销承诺必须拒绝。

### 5.4 PhysicalEvidence

包含事实类型、实体身份、观测时间和来源、会话与模型来源、适用状态版本、误差/有效性条件、验证器版本和审计引用。需区分“未观察到”“确认不存在”“状态不确定”。

holding 被撤销时，依赖它的搬运阶段及后继可行性证书应失效。控制器到位、夹爪开度或物体高度变化单独均不足以证明完整技能成功。

### 5.5 FailureReport

至少记录：失败阶段、类别、是否搜索耗尽、候选及约束诊断、是否发送运动、执行终态、停止证据、物理事实、保留资源、受影响操作以及可允许的恢复动作。

## 6. 阶段事件调度

主状态：

```text
WAITING → READY → COORDINATING → COMMITTED → EXECUTING
                                              ↓
                                          VERIFYING → SUCCEEDED
```

另有 BLOCKED、FAILED、CANCELLING、UNCERTAIN 等分支。状态转换由有身份、时间与序号的事件触发，不使用轮流执行技能的隐式流程。

资格条件：真实前驱已确认完成 + 新鲜物理前置条件有效 + 资源能安全取得或移交。READY 表示可以申请，不保证立即运动。

事件至少包括：PhysicalFactConfirmed/Revoked、PhaseVerified、LeaseTransferred、MotionCommitted/Completed、TrackingDeviation、CancelConfirmed、SceneInvalidated、ReservationWindowAvailable。

事件处理要求：

- 重复事件幂等；旧 attempt 或旧 revision 不能完成新阶段。
- 晚到结果只能补充其原执行记录，不能覆盖当前状态。
- operation 只有在其必要阶段、物理验收及安全义务满足后才能成功。
- 支撑等持续事实可跨 operation 终态保留，租约须显式移交或按终止策略解除，不因成功自动释放。
- 失败传播到真实依赖及冲突关联范围；只有共享世界或执行状态不可信时扩大保护范围。
- 调度需公平性、等待原因和死锁诊断；不能为解除死锁擅自释放支撑或接触资源。

## 7. 运动权限

建议权限类别：

| 类别 | 语义 |
| --- | --- |
| FREE_TO_COORDINATE | 没有必须保持的接触，可在授权范围内规划避让 |
| CARRY_CONSTRAINED | 可移动，但须保持抓持、载荷与姿态约束 |
| CONTACT_PRESERVING | 必须保持支撑/接触，仅允许契约声明的自由度 |
| EXPLICIT_HOLD | 显式要求保持关节或位姿 |
| IN_FLIGHT | 已有轨迹承诺，变更必须经过替换事务 |

权限可组合为约束，不应只靠单一枚举覆盖复杂状态。无事实或权限不明时保守保持。空闲不是无限运动授权；让路运动也必须有轨迹、预约、执行证据。支撑也不总等于固定全部关节，但缺少可信接触模型时不得自行放松。

## 8. 在途轨迹与联合重规划

### 8.1 时空预约

每条承诺保存 q(t)、速度及必要加速度、有效时间、机器人/夹爪/携带物包络、跟踪和时钟误差、观测不确定性、安全停点、停止占用和可替换区间。

比较同一时间的占用，不把其他机械臂的当前姿态外推成静态障碍。离散检测须有保守扫掠包络或经论证的插值误差界，不能只检查稀疏轨迹点。

时间参数化后重新验证联合安全性。夹爪开合会改变包络和持有关系，也须在阶段与预约中表示。

### 8.2 新请求接入顺序

1. 与已承诺轨迹不冲突：独立接纳。
2. 寻找安全开始窗口：只调整新请求时序。
3. 改新请求路径或候选。
4. 让有权限的空闲臂避让。
5. 重规划冲突组的可修改后缀。
6. 安全停止后联合重规划。

不默认重规划全部机械臂，也不默认等全部空闲。优先级规划只作为快速路径；失败时允许换序或联合求解，不宣称问题无解。

### 8.3 替换事务

准备阶段：冻结已执行及不可撤销前缀，选取切换点，生成后缀，验证位置/速度及控制器要求的加速度连续性，准备新预约与执行资源。

提交阶段：以相关世界版本、轨迹 revision、租约与 epoch 为条件进行原子提交；必须有足够提前量，切换点已错过则作废。关联机械臂准备失败时不得部分启动新方案。

提交前失败继续旧的有效承诺；旧承诺已因物理变化失效时进入验证过的停止或接触安全保持方案，不能盲目继续。执行异常造成状态不确定时保留所有相关资源并封锁受影响预约。

初期控制器若不支持可信的在线替换，使用安全边界停止后重新测量、规划与提交；不得为追求平滑拼接牺牲证据。

### 8.4 时钟与重启

运动时间绑定同一会话时钟域，调度超时使用单调时钟。仿真暂停、时钟跳变、过期观测和通信失联必须有显式处理，不能假定时间仍正常推进。

进程重启后旧所有权不可直接恢复使用：提升 epoch，先核对物理状态和未终止动作，未知资源隔离，恢复完成后才允许新承诺。

## 9. 场景、物理事实与租约

场景不使用“任何观测变化都使全局计划失效”的单一版本。至少区分：

- 世界几何/拓扑版本；
- holding、supporting、connection 等物理事实版本；
- 带时间戳的机器人测量状态；
- 已承诺轨迹 revision；
- 接触权限版本。

规划基于一致快照；提交检查其实际依赖；执行监测偏差。正常预测内的运动由轨迹承诺表示，意外物体移动、滑移和拓扑改变撤销相关证书。优化为局部失效前，允许保守扩大失效范围，但不得删除版本门禁。

场景写入需要事务一致性，不应把整个 planning scene 排他锁持有到整段运动结束。模型 attached 状态不是 holding 的物理证明；附着必须来自实测证据，脱离也须有依据。

租约移交需同时满足前驱证据、接收阶段绑定、物理义务连续和场景复核。超时、任务取消、进程死亡或租约到期均不意味着物理资源已释放。

## 10. 面向后继的 PICK

### 10.1 绑定

从外部依赖图与对象关系绑定具体后继 operation，而不是仅使用 `successor_skill`。读取 PLACE_UP / PLACE_DOWN 目标、结构接口、支撑关系、接近/插接/释放/退离通道。

分支存在时不能静默选一个后继：明确分支已选定，或验证所有仍可能被要求的后继；无法定义时报告计划歧义。

### 10.2 抓前预测

技能产生抓持候选域。通用可行性评估使用抓持变换：

`T_world_tcp = T_world_object_target * T_object_tcp`

检查抓取接近、闭合、抬升、搬运、后继预接触、插接、张爪及退离，而不只检查最终 TCP。

先满足硬约束，再评价工具净空、关节裕度、稳定性、后继代价及其他任务影响。工具包络必须包含张爪过程、实物偏置及不确定性。合法接触只限契约中的指定接口；不能靠增加 allowed collision pairs 规避工具干涉。

### 10.3 抓后复核

通过实测物体/TCP 关系、夹爪反馈、跟随证据和误差范围更新抓持 revision。后继可行性证书绑定该 revision、具体目标及相关世界版本。

预测失效时重新规划；仍不可行则安全保持并报告上层。不自动插入未经授权的重抓或放回。可行性证书是预测，不是后继物理完成证明。

## 11. 后端与失败诊断

MoveIt/OMPL 作为初期可替换后端保留，避免同时重写模型、碰撞和求解基础；它们不定义平台能力上限。

| 子问题 | 选择与理由 |
| --- | --- |
| 目标可行性 | 独立 IK、约束采样和碰撞诊断，先分清目标问题 |
| 自由空间路径 | OMPL 基线及可选其他搜索/优化后端，共用问题与验收契约 |
| 在途协调 | 平台时空预约、时间窗与冲突驱动重规划，不能寄托于单次 MoveGroup |
| 紧耦合双臂 | 联合构型空间或时空轨迹优化，保留互相让路自由度 |
| 接触阶段 | 受约束运动与相应接触控制接口；若硬件不具备所需能力则拒绝，不伪装支持 |
| 最终验收 | 与求解后端解耦，换后端不降低安全标准 |

失败至少分为：

1. GOAL_IK_OR_CONSTRAINT：IK 残差、关节限位、候选覆盖与约束诊断。
2. GOAL_COLLISION：碰撞对、距离、候选及状态。
3. PATH_SEARCH_EXHAUSTED：有合法目标，但预算内未找到路径。
4. POST_VALIDATION：轨迹净空、连续性、时空冲突或场景变化验收失败。
5. EXECUTION：控制器拒绝、跟踪偏差、同步或取消失败。
6. PHYSICAL_VERIFICATION：滑移、未连接、未释放、支撑丢失或未完成退离。

保留 UNKNOWN/SEARCH_EXHAUSTED，不把采样失败当数学无解。不通过单纯增加超时或允许非预期碰撞来宣称问题解决。

## 12. 迁移计划

不做单纯文件搬迁；按职责提取并保持可回归的垂直切片。

| 阶段 | 工作 | 出口标准 |
| --- | --- | --- |
| 0 | 固化现有证据、失败分类和安全不变量 | 历史目标/验收失败可区分，不改运行行为 |
| 1 | 定稿新输入、阶段、事件、承诺与证据契约；建立架构测试 | 纯离线 schema、图和导入边界测试通过 |
| 2 | 建立唯一执行入口，迁移 prepare/start/cancel；暂串行 | 新路径无技能直发命令，所有动作有 owner/epoch |
| 3 | 迁移 PICK、PLACE_DOWN、PLACE_UP、SUPPORT_TOP、WAIT 阶段程序 | 保留各自物理证据，不以控制器结果替代 |
| 4 | 事件调度与资源移交，先支持安全边界和无冲突并发 | A→C、B 独立时 C 不因 B 未完成而被层阻塞 |
| 5 | 时空预约、新请求接入、让路与冲突组重规划 | 不把在途臂静态化；时间窗及保守停止验证通过 |
| 6 | 后继抓取预测与实测复核，联合求解后端评估 | 能抓不能放的候选被拒绝；抓后偏差触发复核 |
| 7 | 有条件引入在线后缀替换；清理旧入口和桥接 | 连续性、原子提交、陈旧结果和取消故障测试通过 |

阶段 6 的候选契约设计应在阶段 1 就纳入，避免后续返工；具体求解优化可后置。

现有模块处理：

- `src/rocobrick/piper_run.py` → `src/rocobrick/app/`；更新 CLI 配置，不长期保留顶层转发壳。
- `src/rocobrick/robot_operation_plan.py` → `src/rocobrick/task_plan/`；新输入直接采用新语义。
- `src/rocobrick/multi_robot_scheduler.py` → `src/rocobrick/scheduling/`；保留安全机制，替换层循环。
- `src/rocobrick/unified_runtime_facade.py` → 分别提取到 coordination、world、execution，而不是改名成新万能 facade。
- `src/rocobrick/coordinated_execution.py` → `src/rocobrick/execution/`。
- `src/rocobrick/coordinated_ros_runtime.py` → `src/rocobrick/adapters/moveit/`。
- `src/rocobrick/ros_moveit_runtime.py` → 按规划、场景、观测和控制职责拆分，禁止整块搬入 adapter 后继续混合职责。
- `src/rocobrick/execution/operation_state_ledger.py` → 物理事实归 world，与调度租约和执行所有权分离。
- `src/rocobrick/skills/pick.py` 等 → 阶段程序、几何、候选与验证子模块；删除直接发运动的责任。

迁移桥接必须登记适用范围、删除条件和对应测试。新旧模式会话级互斥，禁止两套执行器同时控制同一机器人。同步更新 AGENTS.md 中的生产链、目录与调度契约，但这是未来实施内容，本次不修改。

## 13. 测试矩阵与验收

测试目录按职责镜像：`tests/task_plan/`、`tests/skills/`、`tests/scheduling/`、`tests/coordination/`、`tests/planning/`、`tests/execution/`、`tests/world/`、`tests/adapters/`；另设 `tests/architecture/` 和 `tests/integration/`。迁移已有测试，不建立重复测试体系。

| 场景 | 验收标准 |
| --- | --- |
| A→C，B 独立 | A 物理完成且 C 条件满足后可提交申请，无整层屏障 |
| 显式多前驱或同步 | 任一必要前驱未确认时不得推进 |
| 两个 PICK、PICK+PLACE_DOWN | 具备安全条件时控制执行区间确实重叠，而非仅 Python task 并行 |
| 空闲臂挡路 | 在授权与安全范围内可让路，不默认 maintain_pose |
| 支撑/接触臂挡路 | 不得为了成功规划破坏持续义务 |
| 未来轨迹相撞 | 当前姿态不冲突也必须识别未来冲突 |
| 路径相交但错峰 | 安全时间窗存在时可接纳，不永久排他占用空间 |
| 携带物或张爪扫掠 | 计入时空包络及误差，不只检查机器人连杆 |
| 轨迹后缀替换 | 不改已执行前缀；切换连续；过期切换点不得提交 |
| prepare 部分失败 | 不部分启动新联合承诺；保留旧有效方案或安全处理 |
| 晚到/重复结果 | 不重复运动，不完成错误 attempt 或 revision |
| 取消失败、失联、重启 | 保留未知资源，epoch 隔离，恢复前不得新发运动 |
| 场景或抓持变化 | 相关计划/证书失效，不能提交陈旧结果 |
| 计划成功但滑移/未连接/未释放 | 物理失败，不能发布 assembled 或成功依赖事件 |
| 可抓不可放 | 抓前拒绝候选；抓后实测偏差导致重新评估 |
| 死锁与饥饿 | 等待原因可审计，有明确仲裁/上报，不私自释放接触资源 |
| 新规划后端 | 通过同一验收与证据契约，不绕过安全门禁 |

保留的现有回归基础：

- `tests/test_successor_pre_phase_scheduler.py`：物理连续性、失败保持租约、移交验证。
- `tests/test_pick_candidate_selection.py`：候选重试、no-dispatch 证明、场景变化与预算。
- `tests/test_coordinated_execution.py`：同步启动、prepare 失败、取消扇出与不安全终态。

架构测试至少保证：skills/scheduling 不导入 ROS、MoveIt 或 controller 客户端；核心不依赖 adapter 实现；contracts 不反向依赖业务；无循环依赖；只有执行适配链可发运动；无遗留旁路。

验证层次：纯函数/契约测试 → 假时钟事件交错与属性测试 → 适配器离线集成与故障注入 → 经单独授权的仿真 → 经单独授权的实机。普通测试不得启动 Isaac/ROS 或发运动。

定量安全阈值应来自标定、控制器能力、跟踪误差和系统时延测量，不在设计阶段拍定。必须配置并审计：净空裕度、误差膨胀、同步容差、停止包络、观测新鲜度和提交提前量。

性能指标可记录 makespan、无依赖等待、规划延迟、重规划次数、证据失效率及资源等待；优化前提是无越权运动、无双重所有权、无陈旧提交、无不安全资源释放。测试通过不等于实机安全认证。

## 14. 待实施前确认与技术探针

不阻塞契约设计，但决定后端和实现细节：

1. 当前控制器是否支持可靠的未来时间启动、后缀替换和取消终态；不支持时采用安全停点拼接。
2. 仿真及未来硬件可提供哪些接触、夹爪与物体观测，如何定义 holding/supporting 的新鲜度与误差。
3. 外部同步是否只需前驱屏障，还是需要同时到达/同时接触；后二者需额外时序契约。
4. 上层是否允许独立任务在另一任务失败后继续；默认只有可证明不受影响的关联外任务可继续。
5. 连续碰撞/扫掠验证与可替换求解后端的具体实现，通过离线基准评估后选择，不预设某库解决全部问题。

本文仍是目标架构；第一批离线实现见实施状态文档。上述运行时探针尚未执行，也不表示已经实现完整新架构。
