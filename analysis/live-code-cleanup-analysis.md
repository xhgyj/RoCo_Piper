# 启动链路活代码与历史数据清理：全面分析

## 结论和范围

本轮按“只保留正式启动链必须用的活代码，删除死代码与无用 runs”的目标做只读分析。分析对象是远端 /home/mzictt/RoCo_Piper，HEAD 14f1c4b，分支 rebuild/piper-isaac-moveit。没有删除、移动、改 ignore、改源码、改配置、改原始证据、暂存、提交、构建或启动仿真。本报告最初保存于 GUI 本地镜像 analysis/；为新会话交接，现已复制至远端 docs/live-code-cleanup-analysis.md。分析阶段未改远端代码；交接阶段仅新增文档。

推荐保留整个正式 operation-plan 执行平台，不把当前单个 Task C 未走的技能当死代码。正式能力与旧 AGENTS 描述存在滞后，以实际源码和启动命令为证据。

证据等级：
- A：正式根、跨进程调用或启动文件直接依赖，必须保留。
- B：不在正式闭包，且有独立旧链路/测试消费者证据；可退役候选，需同步迁移或删除对应旧接口测试。
- C：静态未达或同名遮蔽，较高优先级死代码候选；仍须检查外部接口、源码扫描及删除后验证。
- D：运行历史/缓存/录屏产物，不是代码；按输入与证据价值分类清理。

## 1. 正式多进程启动链

```text
pyproject.toml:93 piper-run = rocobrick.piper_run:main
  -> 外部 robot-operation-plan / structure / config 输入与 TCP 授权
  -> MultiRobotIsaacMoveItSession.start
     -> verify_plan_robots
     -> P1A DDS 前提证据
     -> 必需文件检查
     -> piper_l_model_audit 子进程
     -> generate_dds_profiles 子进程
     -> isaac-ros-env.sh + Python + isaac/p05_host_bridge.py
        -> runtime_host / runtime_host_inputs / isaac_scene / BrickSim
     -> docker run roco-piper-ros:humble
        -> colcon 必要时构建 ROS 包与 C++ sampler
        -> dual_arm_move_group.launch.py
        -> robot_state_publisher / dual_joint_state_aggregator / MoveGroup
        -> ros2_control + arm/gripper controller
     -> docker exec Python -m rocobrick.ros_session_entry
        -> 一次性请求 或 persistent ros_session_worker
  -> UnifiedDualArmRuntimeFacade + 每机器人 adapter / ledger
  -> SkillOperationExecutor + OperationPlanRunner + MultiRobotScheduler
  -> 按 operation plan 分派技能
  -> stop / cancel / cleanup / 终态证据聚合
```

关键源码证据：piper_run.py:792-826、1625-1650、1722-1748、1811-1890、1913-1940、1969；runtime_session.py:2234-2247、2466-2505、2591-2622、2735-2801。所有 src 路径均以 src/rocobrick 为前缀。

Task C 的 type1-c-scaled2-place-up-robot-operation-plan.json 是三层 PICK(piper0)→SUPPORT_TOP(piper1)→PLACE_UP(piper0)，但正式平台仍有 WAIT、TRANSIT、PLACE_DOWN、HANDOVER 分支。不能按这个任务的单条轨迹裁剪全部平台。HANDOVER 使用 DirectSimulationHandover；部分 handover_* 也是共享持物、接触、ledger、prepared-motion 依赖，不能按名字整组删除。

## 2. Python 活代码边界

src/rocobrick 有 137 个物理 Python 文件：135 个规范模块，加被同名包遮蔽的 execution.py 和 skills.py。

考虑包 __init__、函数内导入、from package import 子模块：CLI 根保守闭包 105 个规范模块；再加入 Isaac host 根后为 118 个。此数字是语法上的可能依赖，不是 118 个模块所有函数都被业务执行，也不是动态覆盖率。ROS worker、launch、C++插件、路径字符串需另计。

### 2.1 闭包外 17 个规范模块

下表完整表达“135 规范模块减去闭包外17 = 118 保守可达模块”。均相对 src/rocobrick。

| 模块/文件 | 已确认消费者/归属 | 判定 |
| --- | --- | --- |
| bricksim_backend.py | run/pregrasp_v2.py:17；test_bricksim_backend.py | B 旧回放链，非正式链 |
| trajectory_execution.py | bricksim_backend.py:15、pregrasp_v2.py:27；自身测试 | B 随旧回放链退役 |
| planner_client.py | pregrasp_v2.py:21；test_planner_contract.py:6 | B 旧 gRPC 客户端 |
| planner_contract.py | planner_client.py:12、旧脚本和测试 | B 旧协议 |
| generated/__init__.py、planner_pb2.py、planner_pb2_grpc.py | 旧 client、proto 工具/测试 | B 旧 gRPC 生成代码组 |
| cli.py | rocobrick-validate entrypoint、run/validate_plan.py | B 仍有独立公开预检入口，不是无人使用；退役入口才删 |
| scheduling/__init__.py | 空/兼容包命名空间，未见正式闭包消费者 | C 需核对外部 API |
| session_phase_backend.py | test_session_phase_backend.py:27；转发下项 | B 旧 phase backend 兼容链 |
| skills/phase_backend.py | 兼容转发及测试，内部导 skill_execution | B 旧 phase backend |
| skill_execution.py | phase_backend.py:83、test_skill_execution、test_skills_programs | B 兼容导入；实现 skills/execution.py 仍是活代码，不随转发一起删 |
| skills/handover.py | test_handoff_contract、test_handover_runtime_safety | B 旧 HandoverHandler，不同于正式 DirectSimulationHandover |
| skills/handover_waypoints.py | exploratory 文档，未见 src/tools/tests/planner_ws 导入 | C 高优先级未接入实验代码候选 |
| stap_schedule.py | test_stap_schedule.py | B 旧 STAP 排程编译路径，正式外部 operation plan 不应被重排 |
| supervision.py | test_supervision.py | B 旧独立抽象，不代表可删新 runtime 的 safety/cancel 实现 |
| verification.py | test_verification.py | B 旧独立验证抽象，不代表新证据门禁可删 |

未达不是自动授权删除。闭包外模块可能有独立 CLI、测试、旧脚本和对外消费者。清理应退役整条旧功能，而不是留着旧测试再补假的兼容 stub。

### 2.2 包遮蔽与仍在用的兼容转发

已用 find_spec 验证 rocobrick.execution 与 rocobrick.skills 均解析到各自包的 __init__.py，而非同名 .py。execution.py、skills.py 是优先删除候选 C；其注释称保留供静态兼容，仍需验证 AST/源码扫描测试与外部工具。

不能一刀切删除所有转发：
- robot_operation_executor.py：piper_run.py:1411 正式使用。
- structure_scene.py：Isaac host:2603 使用。
- skill_targets.py：TRANSIT adapter:21 使用，含私有 quaternion helper 转出。
- transit_adapter.py、operation_state_ledger.py：先逐个迁移调用者才删。

__init__.py 隐式导入也是裁剪阻力：根包:3-5 导 contracts/scheduler/task_facts；skills 包导 programs/execution/targets；execution 包导 core。可能只是公共 API 耦合，并不代表业务执行这些旧函数。应单独拆除公共 API 副作用后重新算闭包，不能直接删被它们导入的文件。

## 3. 非 Python 文件同样属于生产活链

必须保留：
- isaac/p05_host_bridge.py；tools/isaac-ros-env.sh、piper_l_model_audit.py、generate_dds_profiles.py。runtime_session.py:109-112 固定路径，:2359-2363检查存在。
- config/dds/fastdds-rootless-tcp.xml.in、fastdds-container-mesh.xml.in，profile generator 生成每会话输出。旧 .runtime/dds 固定文件不等于正式输入。
- artifacts/phase1/dds-multiprocess-gate.json：正式 start:2235 调 _verify_p1a_prerequisite，:2333-2353 强制读取并要求全部 checks=true；不是仅测试报告。长期应考虑迁成明确环境验收契约，但清理不能直接绕过。
- third_party/manifest.lock.yaml、锁定 vendor 及 Git 对象：model audit:329-350 查询 HEAD/gitlink，:404-448 用 hash-object/rev-parse 校验 blobs，故 .git 在当前审计实现里不是可删除缓存。
- robot_assets/piper_l 的运行 URDF/USD/configuration：桥:126和配置直接用，audit与vendor逐层比字节。“旧vendor仅参考”只指不是版本源，不表示运行无人读。
- config/calibration、artifacts/model-audit 的 TCP绑定、校准与模型审计输入；xacro会绑定校准哈希。
- ros_ws/src/rocobrick_piper_l_moveit_config 的 launch、双臂配置生成器、config、xacro、meshes、scripts、C++ sampler头/源、constraint_sampler_plugins.xml、CMake/package。
- ROS vendor agx_arm_description 的上游 Xacro/URDF/meshes 递归闭包。

正式 launch 是 dual_arm_move_group.launch.py，不是旧 dual_move_group.launch.py。它动态加载 dual_robot_config.py/dual_arm_config.py，启自定义 rocobrick/CleanReferenceSamplerAllocator。只保留 Python 会让构建/规划失败。CMake:4-17 构建导出 sampler，:20-23还安装旧fake controller脚本，删除旧脚本需同步构建清单。

Isaac Kit ROS2 extension虽禁用，环境脚本仍从其 humble/lib、humble/rclpy 目录加载；不能据此卸除安装依赖。

重要修正：sim vendor 的 URDF 子模块确实未初始化，但 model audit:654-660明确仅记录且不使用它。它是全上游恢复完整性问题，不能作为“正式启动一定缺必需文件”的证据。ROS侧 URDF 子模块才被实际模型链使用。

## 4. 旧链路与工具删除候选

| 分组 | 分级与理由 | 删除前同步工作 |
| --- | --- | --- |
| docker/planner.Dockerfile、planner-entrypoint.sh | B 高置信旧独立 planner；只构建/启动 planner_ws；正式 ROS 契约 test_ros_runtime_contract.py:101-107 明确禁止 grpc/planner_ws/50051 混入 | 宣告退役旧 gRPC 接口并检查外部部署 |
| planner_ws、proto、planner generated/client、replay_planning_request、generate_python_proto | B 旧服务闭环；正式闭包不需要 | 退役相关协议/资产/模型生成测试；避免误删共享运行资产 |
| run/pregrasp_v2.py、smoke_scene_v2.py、audit_piper_usd.py | B 历史实验/审计入口非正式根 | pregrasp还被TCP历史证据哈希测试读取；先处理旧溯源测试和保留原始归档 |
| run/demo.py、main.py | B 上游示例，导入目前不存在的 env/utils/policy 路径，正式链不用 | README仍宣称main为比赛评估入口，需决定放弃此接口并修说明 |
| run/validate_plan.py、cli.py | B 独立预检入口，非死而是可退役功能 | 同步project.scripts/文档/测试 |
| 旧dual_move_group、move_group、stage4_fake launch/controller | B 条件候选，非当前组合双臂链；旧single session仍引用move_group，CMake仍安装fake | 先删/拆single旧会话分支并更改CMake、测试，再clean colcon build |
| tools/*probe*、smoke*、dds_type_matrix*、ogn_api_probe | B 诊断开发工具，而非正式每次启动必需 | 不能拿“测试import”认定生产必留；决定是否保留诊断工具集及对应测试 |
| tools/sim_collision_hulls.py、task_c_support_station_search.py | B 生成/研究工具，与生产加载的clearance hull JSON分离 | 生产gate/hulls必须保留；生成可复现流程可转维护区 |
| isaac/calibrate_piper_l_tcp.py、publish calibration等 | B 校准维护工具，不是每次正式启动 | 生产可极简不带工具，但若仍维护校准建议单独保留工具包 |
| scene_snapshot_collector.py.orig | C 备份副本，无正式引用证据 | 比较当前差异并归档，勿把未合入改动当垃圾 |

诊断代码不是因为“还有测试”就永远保留；应让测试服务未来保留的产品能力，退役能力时可删除其专属旧测试。新平台的安全测试不能因失败随便删除。

## 5. 依赖包裁剪：只能列候选，不能直接改锁

对第一方 src/isaac/tools/run 的 Python AST检查，openpi、lerobot、isaaclab、chex、torchvision、torchaudio、huggingface_hub 未发现直接 import；torch同样未见第一方直接导入。gRPC/protobuf直接消费者集中旧planner链。

这支持在退役旧链后重新审查 pyproject 中的历史重依赖，不证明可直接卸载：BrickSim/Isaac及其传递依赖可能要求这些包。必须在独立环境解析依赖树、生成新锁并运行真实启动验收，不能在当前 .venv删包来验证。numpy、BrickSim、Isaac、ROS、pinocchio等已有明确代码引用。

## 6. runs：生产不依赖历史输出，空间收益集中在frames

本轮快照：2288顶层目录、76622文件，逻辑字节28,863,201,682（26.88GiB），allocated29,057,630,208。与前轮统计不同，按当前只读遍历结果报告。

| 类型 | 逻辑字节 | 结论 |
| --- | ---: | --- |
| PNG | 26,443,531,136（91.6%） | 最大释放空间候选，主要录屏frames |
| JSON | 2,091,559,618 | 含输入、结果、完整执行证据，需按run分类 |
| MP4 | 188,474,718 | 可选展示/复盘，不证明成功 |
| log | 85,109,839 | 按关键故障选留 |
| 1722个piper-run-*目录 | 合计98,874,030 | 数量多但占用小，非优先空间目标 |

最大8目录及逻辑字节（均runs/下）：

| 目录 | 字节 | 记录状态/原因 |
| --- | ---: | --- |
| task-c-original-view-sharp-20261014-f73b20 | 6873857377 | EXECUTION_FAILED，insertion_step_budget_exhausted |
| task-c-overview-sharp-20261014-54ae8b | 6262399071 | EXECUTION_FAILED，insertion_step_budget_exhausted |
| task-c-pullback-rerecord-20261014-f8e231 | 3212733658 | EXECUTION_FAILED，grasp_slipped |
| task-c-continuous-up-headed-20261014-b372e0 | 3136769986 | EXECUTION_FAILED，unload_up证据失败 |
| task-c-pullback-headed-20261014-128fe7 | 2443934412 | EXECUTION_FAILED，readiness场景freshness |
| task-c-pullback-headed2-20261014-d631a4 | 2442648169 | EXECUTION_FAILED，readiness场景freshness |
| task-c-pullback-headed3-20261014-983b61 | 1557531659 | EXECUTION_FAILED，抓取到达姿态不合格 |
| task-c-overview-headed-20261014-9e52a1 | 945935114 | EXECUTION_FAILED，insertion_step_budget_exhausted |

2221个result.json聚合：1709 INVALID_INPUT、373 EXECUTION_FAILED、29 EXECUTED、28 RUNTIME_START_FAILED、22 EXECUTION_NOT_IMPLEMENTED、12 PREFLIGHT_ONLY、48其他旧schema。29个EXECUTED具有记录中的passed/evidence_complete/execution_safe/tasks_complete/motion_sent/retreat_clear=true，但只是历史字段，不是本轮复验。退出0的PREFLIGHT_ONLY不能当执行成功。

### 6.1 必须分清三类保留理由

**生产硬依赖：**未发现src/isaac/config正式代码硬编码打开历史runs输入。runs是新证据输出根，不是服务启动包；用户选定计划若仍放runs，需要迁为正式输入后更新命令，而非保留整树。

**仅测试硬依赖：**目前3份计划共10,267字节，已有逐字节fixture：compound generated-plan（3383B）、scene-smoke input/plan（3382B）、v2 robot-operation-plan（3502B）。test_scene_manifest_artifact.py:23-28、test_robot_operation_plan_binding.py:37-44通过分段Path读取；test_tcp_structural_binding还按artifact inputs.path读取。这是迁移测试/保持历史provenance可复核的问题，不是生产要求。

**审计追溯：**TCP binding artifact含compound计划、tcp-closed-loop/events.jsonl与approach-planning-request.json历史路径。piper_run只取pinned_files/evidence重建TCP判定，不打开这些历史source/plan路径；保留原证据属于可审计性，不是当前启动硬要求。迁归档应保留原字节/哈希，不篡改不可变artifact。

### 6.2 建议删除/归档策略

1. 第一批空间候选：PNG frame序列，选留MP4/少量关键帧与相应结果证据。只删视频帧不影响正式运行；仍应确认无在用录制进程和展示需求。
2. 第二批：1709个INVALID_INPUT/重复预检，按失败原因+输入哈希+代码版本分组，各留代表，不按目录名盲删。
3. 第三批：旧mode-b/startup/direct-transit/camera/DDS诊断，按缺陷类别留最小证据，旧输出XML/生成URDF/SRDF可归档。
4. 历史成功及关键安全失败另作精简档案，不混称垃圾；单留result不足以复核cancel/terminal/cleanup。
5. 迁完3份测试输入和必要provenance闭包后，生产仓库可以不带历史runs，仅保留创建新证据目录的约定。

每个保留案例最小建议：原命令/argv+env、原始输入及SHA、配置/模型/场景/结构版本、代码版本与dirty信息、provenance/preflight/result/events、operation-plan-run、cancel/terminal/cleanup证据和关键日志。真正保存的launch.sh仅发现两份pick-handover脚本，需连输入闭包一起保留，不单留悬空shell。

## 7. 后续实际清理应如何验收

先生成逐文件保留/删除manifest，再在仓库外副本执行，不从原树试错：

1. 声明保留的产品接口：正式operation-plan全技能平台；是否退役旧grpc/validate/demo/single-session。
2. 分组退役旧链及专属旧测试；不删共享模型/校准/安全门禁。
3. 对仍用兼容转发迁移消费者；拆包初始化旧API隐式耦合，再重算闭包。
4. 迁固定测试输入和phase报告，保留字节/hash和provenance。
5. 不使用现存install：在清理副本做干净colcon构建、Python入口/子进程/launch配置验收；必须包含C++plugin。
6. 运行保留能力的测试；前轮2557通过/27失败/2skip只是质量基线，不用删失败来伪造全绿。
7. 获准后在唯一新run目录做真实启动、选定技能、安全取消和cleanup验收。动态覆盖只能证明测到的分支，不可凭一次Task C运行删除其余受支持技能。
8. 审批后再物理归档/删除并形成结构清理提交，功能修复独立提交。

## 8. 证据与局限

本轮源码读取、AST全分支导入闭包、包解析find_spec、字符串/配置/ROS文件引用、历史JSON及磁盘元数据聚合互相交叉核对；没有完整函数级数据流证明，没有真实runtime coverage，没有卸包/裁剪后的clean构建。118闭包仍过近似，未覆盖所有外部用户脚本与vendor内部动态插件。

原远端tracked diff SHA-256在分析前后相同：be575cafd85c3675f3c5d0613a4ad880bf69c59d792898a8b2163f240b5e9539。index diff为空。无删除、移动、改ignore或源码；本报告作为独立本地交付。
