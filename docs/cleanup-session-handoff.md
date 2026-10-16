# 新会话交接：RoCo_Piper 活代码裁剪与 runs 清理

## 1. 用户真实目标与当前授权

用户原话：“我的目的主要是只留下启动链路中必须要用的活代码，删除死代码，包括runs下之前无用的记录，先全面分析。”

上一轮已完成全面静态分析，尚未开始物理删除。本次用户要求整理交接，以便新会话接手。不要把“忽略生成物、保留所有旧测试依赖”当最终目标；应以正式启动能力为中心退役旧链。分析不是删除授权：新会话应先展示逐文件/分组执行范围，确认退役接口，再开始修改。用户尚未明确只保留当前Task C；推荐默认保留正式operation-plan全部支持技能，不擅自收缩到单任务。

## 2. 工作区与先读文件

- 真正执行仓库（Linux SSH）：/home/mzictt/RoCo_Piper；remote workspace ~/RoCo_Piper。
- 已保存连接profile：pmv7y6fcufbp2（4090，mzictt@10.198.61.26）。用remote_ssh_profiles复核上下文，避免误改本地镜像。
- 本地GUI镜像：C:/Users/DELL/.dsh/remote-workspaces/wmv9dam71df0i。
- 分支 rebuild/piper-isaac-moveit；交接时HEAD 14f1c4b。
- 先读根AGENTS.md，再读 docs/live-code-cleanup-analysis.md（最新权威分析）及本文件。AGENTS能力说明部分滞后，不能拿其中旧“planned/fail-closed”描述否定现代码。
- 辅助报告：docs/repository-baseline-20261015.md（时间序列，后段修正覆盖前段）、docs/dependency-recovery-baseline.md、docs/cleanup-review-checklist.md。它们偏保守且早于最新目标，不应覆盖本次按生产闭包裁剪的意图。

## 3. 前会话实际改过什么

在最初基线阶段：追加 .gitignore（缓存、.zcode、ROS build/install/log、runs子项，保留3个历史输入例外）；新增上述3份基线文档；把两份STAP计划按字节复制到 tests/fixtures/assembly_plans，增加README；将 tests/test_piper_run.py、test_tcp_structural_binding.py、test_structure_scene.py、test_robot_operation_plan_integration.py 各改一处路径。原runs输入不变。

后续全面活代码分析阶段未改远端源码/配置/ignore，未移动或删除数据。交接阶段仅新增本文件及完整分析报告到远端docs并提供本地副本。没有暂存或提交，没有reset/checkout/clean，没有uv sync，没有真实Isaac/ROS/机器人运行。

原有dirty功能修改必须保留：最初46个tracked修改/删除（含CLAUDE.md、tools/sync_piper_usd_limits.py原有删除），后来加ignore和四个测试改动后tracked dirty为51项。已有修改包括技能、runtime、ROS模型、测试、uv.lock；不要回滚来获取表面干净。

交接时tracked diff SHA-256：be575cafd85c3675f3c5d0613a4ad880bf69c59d792898a8b2163f240b5e9539；index diff标准空SHA e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855。重新核实，文件可被用户继续修改。

## 4. 正式启动链与硬保留

piper-run = rocobrick.piper_run:main → immutable operation-plan/structure/config门禁与TCP授权 → MultiRobotIsaacMoveItSession →模型审计/DDS→Isaac host→ROS容器组合双臂MoveGroup→ROS持久worker→SkillOperationExecutor/OperationPlanRunner/MultiRobotScheduler→cancel/terminal/cleanup证据。

硬保留隐藏依赖：
- runtime_session.py:109-112启动 isaac/p05_host_bridge.py、tools/isaac-ros-env.sh、piper_l_model_audit.py、generate_dds_profiles.py。
- :2234-2247的start还在:2333-2353强制读 artifacts/phase1/dds-multiprocess-gate.json；它不是仅测试旧报告，不得直接删/绕过。
- 正式 ROS launch 是 dual_arm_move_group.launch.py，不是dual_move_group；动态加载dual_robot_config/dual_arm_config，依赖C++ CleanReferenceSamplerAllocator、plugin XML、include/src、CMake安装项及topic_based_ros2_control。
- ros_session_entry 还有 --serve-dir/worker-token 持久模式，execution/ros_session_worker、ros_worker_transport必须保留。
- robot_assets/piper_l 与两个锁定vendor同时被模型审计用；vendor .git 被查询HEAD/gitlink/blob，不是可直接删的缓存。
- ROS2 Kit extension虽禁用，wrapper仍从其humble/lib和rclpy目录加载。
- clearance模块及piper_l_sim_hulls.json由coordinated_session_entry真实导入，未跟踪不等于垃圾。camera_stand xacro/DAE是真模型依赖。

## 5. 代码裁剪分析结论

137物理Python文件=135规范模块+2同名被包遮蔽文件。包含包初始化、局部import、host子进程后保守可达118规范模块；不是函数级活跃证明。闭包外17具体清单及消费者见完整分析报告第2节。

优先分组候选：
1. 旧gRPC planner独立链：planner_ws、docker/planner.Dockerfile/planner-entrypoint.sh、proto、generated planner_pb2/_grpc、planner_client/contract、replay/generate_proto工具；正式ROS契约明确排斥旧planner混入。仍有旧协议/资产/模型生成测试，退役旧功能时同步处理，不是“测试读就永远留”。
2. 旧pregrasp/BrickSim trajectory playback链；pregrasp还被历史TCP溯源哈希测试读取，先处理归档/测试契约。
3. 旧phase_backend/session_phase_backend、旧skills/handover.py HandoverHandler、stap_schedule、supervision、verification。主要旧测试消费者；不等同新runtime安全功能。
4. skills/handover_waypoints.py未见入边，execution.py/skills.py被同名包遮蔽，.orig备份：高优先候选，检查外部和源码扫描后再删。
5. 诊断probe/smoke、研究搜索、校准工具不是每次运行必需，可拆维护工具集；生成工具与生产加载的JSON资产分开。
6. 旧single-session/launch/fake controller要连runtime分支、CMake安装清单一起退役，不能只删文件。

不能整组删除handover_*：正式DirectSimulationHandover及共享contact/ledger/evidence还在使用。当前Task C只有PICK→SUPPORT_TOP→PLACE_UP，不代表WAIT/TRANSIT/PLACE_DOWN/HANDOVER支持分支是死代码。多个兼容转发仍有正式调用者，须先迁调用。根__init__还隐式拉旧公共API，拆副作用后闭包可进一步缩减。

pyproject的openpi/lerobot/isaaclab/chex/torchvision/torchaudio/huggingface_hub等未见第一方直接import，仅为依赖裁剪候选，未检查第三方传递依赖，禁止直接卸包/删锁。

## 6. runs的结论和坑

只读快照2288目录/76622文件，28,863,201,682逻辑字节=26.88GiB；PNG占26,443,531,136字节=91.6%。优先清录屏frames，比删1722个piper-run小目录（总共98.9MB）收益大。最大8个录屏目录全部EXECUTION_FAILED；结果字段与尺寸完整列表见报告。

2221个result：1709 INVALID_INPUT、373 EXECUTION_FAILED、29 EXECUTED、28 RUNTIME_START_FAILED、22 EXECUTION_NOT_IMPLEMENTED、12 PREFLIGHT_ONLY、48旧schema。历史EXECUTED不是本次复验；PREFLIGHT_ONLY exit0不是成功执行。建议按输入hash/原因/代码版本选代表归档，再清重复记录。

生产未发现硬编码打开旧runs；历史路径只作输出根/字符串，不需带整树部署。以下3份却仍是测试硬依赖，共10267B，均已有字节一致fixture：
- runs/example1-compound-pick-20260915T130000Z/generated-plan.json → tests/fixtures/assembly_plans/example1-compound-pick.json，SHA 02b5f86513f1daaf6763b787c5a791b1acd0a7656e8cf0fb1e5420e623f9cae3
- runs/scene-smoke-20260914T093942Z/input/plan.json → tests/fixtures/assembly_plans/example1-scene-smoke.json，SHA ff2ccfc1ee89286cb21ab94ad96f315785bd8be60599244f29ce743bddb7b92a
- runs/example1-v2-plan-fixture/example1-v2-robot-operation-plan.json → tests/fixtures/robot_operation_plans/example1-robot-operation-plan.json，SHA 07c072749dff371db8d15933af5fbed6bf0b444e4ebbc096d82c8e99f4c22c1a

test_scene_manifest_artifact.py、test_robot_operation_plan_binding.py分段Path残留，test_tcp_structural_binding按artifact路径重哈希；此前只搜runs/连续字符串漏掉过。TCP binding artifact里还有events/approach request provenance路径，正式TCP读取未解引用这些旧源路径。生产依赖、测试依赖、审计归档价值必须分开，别再笼统称“必须留runs”。

选留案例需保留输入原字节/哈希、argv/env、代码dirty版本、result/preflight/provenance/events、operation-plan-run、cancel/terminal/cleanup证据；单留视频或result不足以证明安全。251个tracked历史runs移出index与物理删除是两件事，需单独计划。

## 7. 现有验证、依赖事实与限制

已测：核心176 passed；全量2557 passed/27 failed/2 skipped；Ruff2679问题未fix。27失败集中旧/现HANDOVER几何阶段契约和两个mimic检查。不能因失败就删仍保留能力的安全测试。

无运行产物隔离副本：/tmp/roco-clean-baseline-20261015-7kipxnd1/candidate。含必要模型/vendor/契约报告与3个runs输入，不含原Git、.venv、build/install/log、录屏/缓存。使用原.venv、PYTHONPATH优先副本src，复现2557/27/2；不是fresh install/clean Git checkout/删除候选后验证，也没测真实runtime。
清单 /tmp/roco-clean-baseline-20261015-7kipxnd1/inventory.json：1643项，SHA a712a880e5d82cb97eba869b2daa111d572594aa7d40ade3d9a9408f939aba78。/tmp非持久备份，先查存在。

第三方实测HEAD：sim f8cd8b147c75d59e14f90fb0646770eefa268ed0；ROS b9ad14de2a6eb818a2d206fbd1adba48343662f7；ROS URDF子模块 f6642ce0d7872c686f29c99e9e10cd23d1d49313。许可源SHA与manifest一致。sim URDF子模块1ee5659d02b33b9379fd647a2e5647800d67f0f4未初始化，但audit:654-660明确不用它，不是当前正式启动阻断。sim USD/mesh许可仍未澄清。

## 8. 新会话建议执行顺序

1. 读取最新完整分析和AGENTS，复核Git现状/运行中进程/磁盘，别重复做纯盘点。
2. B5 已确认并实施：validate/demo 入口退役，旧 launch 探针与诊断工具归档到 maintenance/archived_tools/；正式全技能 operation-plan 链保持不变。
3. 生成逐文件keep/remove/migrate manifest，附静态调用证据、关联测试/构建/文档调整。
4. 先在隔离副本分组执行裁剪（需获授权），迁兼容调用和固定测试输入；不要在原树试错，也不靠ignore隐藏死代码。
5. 清洁ROS构建验证C++plugin/launch/xacro及子进程导入，不借用陈旧install遮掩源码缺失。
6. 跑保留能力测试并对比known baseline，再经授权进行真实启动、取消、cleanup验收。
7. 最后归档代表证据、清PNG/重复runs、处理tracked生成物，结构清理与功能修复分开，不擅自提交。

## 9. 工具注意

remote_ssh_*操作真实仓库，本地read/write操作Windows镜像；文档呈现需本地副本+present，远程绝对路径在GUI不一定可开。不要whole-tree sync/push（runs巨大）。远端多行heredoc多次超时且没进入执行，python3 -c可靠；用python3而非python。remote_ssh_grep单文件行号解析偶有错位，引用行号可cat后自行计数；其模式是grep ERE，括号转义规则别用错。枚举未跟踪全树曾慢，优先限定路径。
