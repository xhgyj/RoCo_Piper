# 仓库基线盘点（2026-10-15）

## 状态与边界

这是清理前的可测量基线，不是“干净工作树”或“端到端全绿”声明。执行位置 `/home/mzictt/RoCo_Piper`，分支 `rebuild/piper-isaac-moveit`，HEAD `14f1c4b`。根 README 主要是上游 DexMate/BrickSim 说明，不能作为 Piper 正式启动手册。

盘点时 tracked 1126 项，原有 tracked 修改/删除 46 项，runs 中 tracked 251 项；只读枚举得到 untracked 77084 项，其中非 runs 776 项。runs 有 2270 个目录、约 28G；环境 .venv 约 20G，ros_ws 约 382M。统计是盘点快照，不是删除清单。du 扫描 .venv 的截图目录遇到权限限制，未修权限。

本轮仅新增本文，追加 .gitignore 中 Python/pytest/Ruff 缓存、.zcode、本地 ROS build/install/log 规则。没有删除、归档、reset、暂存、提交、依赖同步或启动 Isaac/ROS/机器人运动；不忽略 runs、vendor、tools、src、tests、artifacts 或模型目录。

## 项目与正式启动链路

Linux x86_64 / Python 3.11 / Isaac Sim 5.1 / BrickSim / ROS 2 Humble / MoveIt。src/rocobrick 是 CLI、输入契约、layer scheduler、技能、MoveIt adapter、session lifecycle；isaac/p05_host_bridge.py 是仿真桥；ros_ws/src/rocobrick_piper_l_moveit_config 是双臂命名空间配置；config/tasks/robot_assets/artifacts 中含必须保留的运行输入与溯源。

正式派发入口只有 `piper-run --robot-operation-plan PATH`。外部 operation plan 不可重排或修补；可结合 STAP/binding/TCP gate 后进入 readiness → layer scheduler → RobotOperationExecutor → evidence/cleanup。拓扑是一个 Isaac host bridge 加一个容纳全部计划机器人的 ROS/MoveIt container session，机器人使用独立命名空间并共享 clock。旧 demo 与 tools probe/smoke 不是正式调度替代品。

历史命令来源：docs/piper-pick-implementation.md:303-309；runs/pick-handover-headed-retry-20261006T111512Z-VgcQEM/launch.sh:5-15。形式如下（示意，不是当前 Task C 已验证复现命令，不要照抄旧输入执行）：

```bash
ROCOBRICK_HEADED=1 PYTHONPATH="$PWD" .venv/bin/piper-run \
  --robot-operation-plan /absolute/path/to/robot-operation-plan.json \
  --structure-goal /absolute/path/to/structure_goal.json \
  --structure-start /absolute/path/to/structure_start.json \
  --config /absolute/path/to/user_config.json \
  --run-dir runs/NEW-UNIQUE-DIRECTORY
```

历史脚本还使用 --scene-manifest、--scene-snapshot-out、--host-instance-id；不能漏掉原始配置、环境变量和输入哈希。当前 scaled2 C 输入线索为 artifacts/scene/type1-c-scaled2-place-up-robot-operation-plan.json 与 tasks/type1_scaled2/C 下结构输入。先固定全部依赖，再生成正式可复现启动说明；本轮没有证实最新运行的完整原始 shell 环境。

真实 runtime 还需 GPU、Docker、roco-piper-ros:humble、可读 DDS profile 和可用端口。tools/isaac-ros-env.sh 使用 Humble bridge、rmw_fastrtps_cpp、默认域 73。全新 checkout 是否可构建/启动尚未验证。

## 历史运行判定

以下路径均在 runs 下，读取其嵌套执行目录 result.json；视频、输入预检通过或 motion_sent=true 都不能替代最终成功。

- task-c-pullback-rerecord-20261014-f8e231 / 20261014T151605.422821Z_79fd5345-dcab-4505-a870-6434888807d5：EXECUTION_FAILED / exit 10，grasp_slipped，未确认向上连接。
- task-c-pullback-headed3-20261014-983b61 / 20261014T151218.184168Z_0938dfdb-3479-4fd7-9b02-bb4a64b41555：EXECUTION_FAILED / exit 10，PICK arrival pose outside target。
- task-c-continuous-up-headed-20261014-b372e0 / 20261014T145418.131627Z_dfdd7eb3-9e9e-438a-9f85-8ffce83d8fc2：EXECUTION_FAILED / exit 10，unload_up runtime evidence failed。

上述三次均 input_valid=true、motion_sent=true、passed=false、execution_safe=false、retreat_clear=false。仅可参考启动与失败证据。

历史 example1 成功参考：pick-cleanup-fixed-20261002T062022Z / 20261002T062022.768769Z_28807fce-6231-40a1-9067-6a4d75325737；place-simple-g-20261002T153845Z / 20261002T153845.619099Z_bf87694d-4499-4593-a8ca-385bd2bed963。对应结果 EXECUTED / exit 0 / passed 和 execution_safe=true，仅证明当时 PICK/PLACE_DOWN，不证明当前 Task C PLACE_UP 成功。

## 清理分类与阻断项

1. **可忽略但本轮不物理删除**：Python/pytest/Ruff 缓存、.zcode、ros_ws/build/install/log。安装树扫描出现过符号链接层数过多，应独立排查，不能据此判定源模型损坏。
2. **runs 先提取 fixture 再处理**：tests/test_piper_run.py、test_structure_scene.py、test_tcp_structural_binding.py 直接读 runs/example1-compound-pick-20260915T130000Z/generated-plan.json；test_robot_operation_plan_integration.py 读 runs/scene-smoke-20260914T093942Z/input/plan.json。后续复制到稳定 fixtures、核对 SHA-256、更新测试后再归档历史数据；251 个 tracked 证据需逐项评审是否移出 index，ignore 不会取消跟踪。
3. **未跟踪但必须保留**：src/rocobrick/clearance 模块、hulls JSON、相关测试；tools 中大量 probe/smoke 被 tracked 测试直接加载；camera_stand xacro 和对应 DAE 被 MoveIt 模型引用；third_party/README.md 与 manifest.lock.yaml 为依赖声明。不能 blanket ignore 或 git clean。
4. **模型/溯源不能随意清理**：robot_assets/piper_l 仍被配置、校准和旧 planner Docker 构建引用；artifacts/model-audit、calibration、scene 与 tasks 是运行输入，不是全体生成垃圾。phase 目录、.orig 备份与旧工具需要逐项引用审查。
5. **vendor 尚未形成可恢复基线**：isaac/vendor/agx_arm_sim、ros_ws/src/agx_arm_ros 是未跟踪嵌套仓库；顶层无 gitlink，clone 顶层不能恢复它们。manifest 声明 ROS commit b9ad14de2a6eb818a2d206fbd1adba48343662f7、sim commit f8cd8b147c75d59e14f90fb0646770eefa268ed0、ROS URDF 子模块 f6642ce0d7872c686f29c99e9e10cd23d1d49313；本轮未核验实际 vendor HEAD/许可哈希，不能写成已验证。sim USD/mesh 许可明确仍待 P0.4 审查。

## 实测质量基线

当前 .venv Python 3.11.15，Ruff 0.15.11。SSH 非交互 PATH 下未找到 uv；本轮直接用已存在 .venv，未下载/更新锁定依赖。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_robot_operation_plan_integration.py tests/test_piper_run.py \
  tests/test_structure_scene.py tests/test_tcp_structural_binding.py
# 176 passed in 4.83s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" .venv/bin/python -m pytest -q --tb=no -p no:cacheprovider tests/
# 2557 passed, 27 failed, 2 skipped; exit 1

.venv/bin/ruff check --statistics
# 2679 errors; exit 1; 未运行 --fix
```

27 个测试失败分布：test_handover_preparation.py 18，test_handover_program.py 1，test_handover_region.py 1，test_handover_scene_planner.py 1，test_handover_staged_execution.py 4，test_host_bridge_scene_state.py 1，test_ros_moveit_runtime_contract.py 1。表现集中在交接几何/阶段契约和夹爪 mimic，不在本次纯整理中修行为。Ruff 主要为文档、长行和导入等存量问题，不通过放宽规则或忽略失败伪造全绿。

## 建立真正干净基线的后续顺序

- 先把原有修改与未跟踪有效源码逐项确认，保留现有成果，不以 reset/clean 获取表面干净。
- 将历史输入迁到稳定 fixture，记录哈希并跑上述 176 项核心测试和全量测试。
- 固定并核验 vendor 获取方式、实际 commit、许可、模型/校准依赖；补齐 clean checkout 所需文件。
- 确认保留的成功/失败证据目录，建立仓库外归档清单和恢复方式，然后处理 runs 与 tracked 生成证据。
- 将源码补齐、结构清理、功能修复分开提交；27 个失败先作为已知基线，不混入无关行为修改。
- 在新 checkout 中验证离线测试与资产恢复；获准后在唯一新运行目录执行真实仿真验收。最终 clean Git、可恢复依赖、测试结果与端到端能力分别报告。

原始 tracked diff SHA-256：8014b4b963052a5f8acd2fba6135633ce458e50594d9c039280fb11645a91889。追加 ignore 后排除 .gitignore 重算一致；index diff 为标准空 SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855。

## 后续整理：稳定 fixture 解耦

已将两份历史 STAP 输入按原始字节复制到 tests/fixtures/assembly_plans，保留两个不同诊断计时的版本，四处测试引用改用稳定路径。来源及 SHA-256 见该目录 README.md；原始 runs 输入未改动。迁移后核心 176 passed，全量仍 2557 passed / 27 failed / 2 skipped，失败项与迁移前一致。tests 中 runs/ 剩余匹配仅历史注释、来源记录和合成 evidence 字符串，无本轮发现的真实文件读取依赖。

因此追加 /runs/ ignore，只控制未来/未跟踪生成证据，未删除数据，未移除现有 251 个 tracked 证据，也未暂存/提交。本节取代前文“尚未迁移 fixture、暂不忽略 runs”的阶段状态；前文保留为初始快照。真正 clean checkout、vendor 恢复与 tracked 证据归档仍待后续验证。

## 后续核验：vendor 与未跟踪报告

已实际核验 sim、ROS vendor 和 ROS URDF 子模块 HEAD 及两份许可源 SHA-256，均符合锁定清单；但 sim 的 URDF 子模块 1ee5659d02b33b9379fd647a2e5647800d67f0f4 尚未初始化（目录为空），因此不能宣称依赖树完整恢复。详细证据、恢复边界及遗漏文件清单见 docs/dependency-recovery-baseline.md。

新发现：artifacts/phase* 下多份未跟踪 JSON 也是 tracked 测试读取的契约报告；不能把所有 phase 目录都当可丢弃日志。后续需分离冻结 fixture 和运行日志后再清理。排除本轮 .gitignore 和四个测试路径后，原有 diff SHA-256 仍为 8014b4b963052a5f8acd2fba6135633ce458e50594d9c039280fb11645a91889。

## 隔离候选基线验证与前述结论修正

已建立仓库外候选副本 /tmp/roco-clean-baseline-20261015-7kipxnd1/candidate（约 783M），排除 Git 元数据、.venv、.runtime、历史 runs、ROS build/install/log、Python/pytest/Ruff 缓存、.zcode 与 .orig。保留有效源码、vendor/模型和契约报告；复用原 .venv 的解释器和依赖，PYTHONPATH 显式优先候选 src。不是新依赖安装或 clean Git clone 验收。

第一次全量测试 2551 passed / 33 failed / 2 skipped，暴露 6 项间接历史路径依赖：test_robot_operation_plan_binding.py 用分段 Path 拼接历史路径，test_scene_manifest_artifact.py 同样分段拼接，test_tcp_structural_binding.py 还从冻结 artifact 的 inputs.path 读取原始文件。先前按 runs/ 连续字符串搜索得出的“已完全解耦”结论不成立。

为不改写冻结溯源输入，当前 .gitignore 改为忽略 runs 子项但显式保留以下 3 份原始输入（不是整个历史目录）：

- runs/example1-compound-pick-20260915T130000Z/generated-plan.json，SHA-256 02b5f86513f1daaf6763b787c5a791b1acd0a7656e8cf0fb1e5420e623f9cae3
- runs/scene-smoke-20260914T093942Z/input/plan.json，SHA-256 ff2ccfc1ee89286cb21ab94ad96f315785bd8be60599244f29ce743bddb7b92a
- runs/example1-v2-plan-fixture/example1-v2-robot-operation-plan.json，SHA-256 07c072749dff371db8d15933af5fbed6bf0b444e4ebbc096d82c8e99f4c22c1a；与现有 tests/fixtures/robot_operation_plans/example1-robot-operation-plan.json 字节相同。

将这 3 份输入按原始字节补入隔离候选后，全量恢复为 2557 passed / 27 failed / 2 skipped（21.79s），与工作区质量基线一致。原始历史证据没有改写或删除，新增 6 项失败没有被 skip/放宽断言。说明其余历史 runs 与构建缓存不是本次离线测试的必要条件；真实仿真与 fresh dependency install 仍未验收。此节取代先前完全解耦/无历史读取依赖的结论。

复测命令：在候选目录执行 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD" /home/mzictt/RoCo_Piper/.venv/bin/python -m pytest -q --tb=no -p no:cacheprovider tests/。临时副本保留供后续验证，不是正式发布基线。

## 本阶段交付边界确认

用户选择“保留已验证候选，先评审清理清单”，不提交、不删除数据。已生成 docs/cleanup-review-checklist.md，列明有效未跟踪依赖、冻结输入、vendor、旧链路和历史证据的后续决策。隔离候选 inventory.json 共 1643 项，SHA-256 a712a880e5d82cb97eba869b2daa111d572594aa7d40ade3d9a9408f939aba78。本阶段以初步认知、可复测隔离候选与清理评审清单为交付；后续物理归档、正式提交及从零恢复验收不在本阶段擅自执行。
