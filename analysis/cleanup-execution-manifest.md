# RoCo_Piper 活代码裁剪执行清单（待确认）

## 0. 授权状态与复核基线

这是执行范围提案，不是删除授权。本轮仅做远端只读复核；未改源码、配置、测试或历史证据，未暂存、提交、构建、启动仿真。沿用两份最新交接分析，不重新盘点全仓。

- 真实仓库：/home/mzictt/RoCo_Piper；分支 rebuild/piper-isaac-moveit；HEAD 14f1c4bcac03f086b54369ce8235791ac6f94ea7。
- tracked dirty 51 项，包含既有功能改动和既有删除；tracked diff SHA-256：be575cafd85c3675f3c5d0613a4ad880bf69c59d792898a8b2163f240b5e9539；index diff 为空，与交接一致。
- 未观察到相关 piper/Isaac/ROS/colcon/录制进程；docker ps 为空。这只是本轮快照，删除前必须重查。
- /home 可用约300G，/tmp 所在分区约900G。旧隔离副本仍存在，但不是本轮裁剪验证，不复用其结论充当验收。
- 历史 runs 仍有251个tracked文件。本轮不重遍历26.88GiB历史树；容量、结果分类沿用报告，执行前生成精确候选清单。
- 三份历史计划与现有fixture重新逐字节核对一致，哈希仍与交接吻合。

## 1. 产品边界：KEEP

默认保留正式 operation-plan 全技能平台，而不是只留 Task C：WAIT、TRANSIT、PICK、PLACE_DOWN、HANDOVER、SUPPORT_TOP、PLACE_UP及其安全拒绝/未实现门禁都不能凭一次运行覆盖删除。

以下为硬保留组（组内仍允许后续有证据的函数级裁剪，本轮不做盲删）：

	src/rocobrick/piper_run.py
	src/rocobrick/multi_robot_scheduler.py
	src/rocobrick/multi_robot_runtime.py
	src/rocobrick/runtime_session.py 的正式多机器人及共享部分
	src/rocobrick/ros_session_entry.py、ros_moveit_runtime.py
	src/rocobrick/coordinated_session_entry.py、coordinated_ros_runtime.py
	src/rocobrick/execution/（除下述明确退役兼容链）
	src/rocobrick/skills/（除下述旧Handler/phase/实验文件）
	src/rocobrick/adapters/、clearance/、plan/
	isaac/p05_host_bridge.py
	tools/isaac-ros-env.sh、piper_l_model_audit.py、generate_dds_profiles.py
	config/dds/fastdds-rootless-tcp.xml.in、fastdds-container-mesh.xml.in
	artifacts/phase1/dds-multiprocess-gate.json
	config/calibration/、artifacts/model-audit/、运行场景manifest及clearance hull JSON
	robot_assets/piper_l/、third_party/manifest.lock.yaml
	isaac/vendor/agx_arm_sim/、ros_ws/src/agx_arm_ros/及当前审计必需Git对象
	ros_ws/src/rocobrick_piper_l_moveit_config/ 的正式双臂链


ROS硬保留明细：

	launch/dual_arm_move_group.launch.py、dual_robot_config.py、dual_arm_config.py
	scripts/dual_joint_state_aggregator.py
	src/clean_reference_sampler_allocator.cpp、include/
	constraint_sampler_plugins.xml、test/test_clean_reference_union_sampler.cpp
	CMakeLists.txt 的sampler构建/导出/安装和gtest项、package.xml
	正式config/、urdf/、meshes/及camera_stand xacro/DAE

必须保留持久worker --serve-dir/worker-token、每机器人动作/取消通道、terminal/cleanup证据。vendor .git 和禁用Kit扩展内仍加载的 humble/lib、rclpy 不当缓存删。DDS前提报告当前被start强制读取，不能为极简化绕过。

## 2. 分批 REMOVE / MIGRATE 清单

所有REMOVE先在隔离候选副本执行；原树回应用单独报告确认。通配分组必须在执行前展开成路径+SHA-256的精确manifest，不直接用于rm。

### B0：固定输入与审计溯源迁移（先决批次）

MIGRATE：

	tests/test_scene_manifest_artifact.py 的PLAN_PATH
	tests/test_robot_operation_plan_binding.py 的EXAMPLE1_STAP_PATH、EXAMPLE1_PLAN_PATH
	tests/test_tcp_structural_binding.py 的历史inputs/source哈希读取

前两者改读已有 tests/fixtures/assembly_plans/ 与 tests/fixtures/robot_operation_plans/ 中对应计划。保留既有四份测试的fixture修改，不回滚重做。

TCP测试使用显式、有限的历史路径→归档对象映射，校验原字节SHA，未知路径失败；不得增加“文件不存在则跳过”的fallback。将需复核的pregrasp原脚本及历史证据作为不可执行归档对象移出生产树，归档索引记录旧路径、哈希、新位置。不可变TCP artifact保留原字节、旧source字符串及分类，不把legacy assumption改成有效校准。纯合成 source 字符串不必重写。

联动：fixture README增加哈希/来源；TCP/scene/binding/plan integration focused tests。B0完成前不得删除相关runs输入或pregrasp原字节。

### B1：旧gRPC与pregrasp/playback退役

REMOVE：

	planner_ws/、proto/
	docker/planner.Dockerfile、docker/planner-entrypoint.sh
	src/rocobrick/generated/
	src/rocobrick/planner_client.py、planner_contract.py
	src/rocobrick/bricksim_backend.py、trajectory_execution.py
	run/pregrasp_v2.py
	tools/replay_planning_request.py、generate_python_proto.sh
	tests/test_planner_contract.py、test_dual_model_generation.py
	tests/test_bricksim_backend.py、test_trajectory_execution.py

MIGRATE而非整删：
- tests/test_assets.py：仅退役旧generate_dual_model测试及失用常量，保留runtime URDF与锁定资产一致性、宽限位拒绝测试；可另补正式双臂生成器覆盖，不能搬运旧0.1358断言。
- tests/test_r2_live_adapter.py：去掉已退役模块的compat/import列表，保留容器Python3.10/StrEnum语义；源码扫描改指向真实包实现。
- tests/test_third_party_manifest.py：README断言改成明确旧planner退役及正式vendor/许可来源，不删manifest/hash/license验证。
- tests/test_ros_runtime_contract.py、test_piper_run.py 的禁导旧planner负向护栏保留。
- README改成正式piper-run链，旧比赛demo/评估描述不再充当当前启动承诺；上游许可/出处保留。

构建：ROS镜像继续保留；不因gRPC源码删除直接删uv锁里的grpc/protobuf，依赖另批验证。

### B2：旧phase/独立抽象退役，保留安全语义

REMOVE：

	src/rocobrick/session_phase_backend.py、skills/phase_backend.py
	src/rocobrick/stap_schedule.py、supervision.py、verification.py
	tests/test_session_phase_backend.py、test_stap_schedule.py
	tests/test_supervision.py、test_verification.py

MIGRATE后REMOVE：src/rocobrick/skill_execution.py；先将 tests/test_skill_execution.py、test_skills_programs.py 导入迁到 rocobrick.skills.execution，两个测试文件及真实实现保留。

条件REMOVE：src/rocobrick/skills/handover.py 旧HandoverHandler；必须先处理以下混合测试，不整组删handover_*：
- tests/test_handoff_contract.py：保留pre-phase/shared snapshot、严格身份/布尔/follow_samples和ledger unknown-state测试；旧Handler ownership/partial-evidence/backend断言迁正式DirectSimulationHandover/contact/ledger。
- tests/test_handover_runtime_safety.py：保留现有dirty +7行功能测试修改。后段三项perform_coordinated_plan/SessionWorker no-dispatch、unsafe-after-dispatch、orderly-stop测试与_planning_fixture原样保留，解除顶层旧Handler import；前段contact、foreign provenance、clearance、cohold/release失败、uncertain ledger、post-approach TCP断言迁现API后才能退役旧API bodies。
- 现DirectSimulationHandover、coordinated_handover、handover_contact/geometry/live和共享ledger/evidence均KEEP。

### B3：遮蔽文件与兼容转发清退

优先REMOVE候选：

	src/rocobrick/execution.py、skills.py（同名包遮蔽）
	src/rocobrick/skills/handover_waypoints.py
	src/rocobrick/scheduling/__init__.py（无消费者再删空目录）

联动 tests/test_r2_live_adapter.py 的静态扫描，不删真实execution/、skills/包。

MIGRATE全部消费者后REMOVE转发：

	robot_operation_executor.py -> execution/robot_operation_executor.py
	operation_state_ledger.py -> execution/operation_state_ledger.py
	transit_adapter.py -> adapters/moveit/transit.py
	skill_targets.py -> skills/targets.py（包括私有_quat_mul/_quat_rotate_vector）
	structure_scene.py -> plan/structure_scene.py

消费者包含piper_run、Isaac host、正式executor、pick/place_down/support_top/place_up、MoveIt transit和对应executor/ledger/transit/target/integration测试；不能只改测试不改生产import。保留现有dirty功能语句，仅做精准导入/patch-target迁移。

根包及skills/execution初始化副作用：MIGRATE为最小明确API，先记录现公共导出及真实调用；不将由旧__init__拉入的contracts/scheduler/task_facts直接判死。闭包重算只针对变化边界，不重做全面盘点。

src/rocobrick/scene_snapshot_collector.py.orig：与当前文件不同，先diff与原字节归档；有独有未合入行为先报告。归档完成后才REMOVE，不直接丢弃。

### B4：旧single-session与旧ROS启动栈

REMOVE（以退役单臂接口为前提）：

	ros_ws/src/rocobrick_piper_l_moveit_config/launch/move_group.launch.py
	ros_ws/src/rocobrick_piper_l_moveit_config/launch/dual_move_group.launch.py
	ros_ws/src/rocobrick_piper_l_moveit_config/launch/stage4_fake_controllers.launch.py
	ros_ws/src/rocobrick_piper_l_moveit_config/scripts/stage4_fake_arm_controller.py

MIGRATE runtime_session.py，精准移除旧CONTAINER_LAUNCH_SCRIPT、RuntimeSessionConfig/ReadinessEvidence/SessionEvidence、IsaacMoveItSession及旧顶层host-time helpers。不得删除整文件或前半段：RuntimeSessionStatus/Error、_is_finite_positive、_with_headed_display、RunOutcome、ProcessHandle/Runner、SubprocessRunner以及路径/DDS/取消/cleanup常量仍被multi正式链使用。多臂同名host-time方法保留。

测试：tests/test_runtime_session.py按single专属用例退役，先保留/迁出共享与禁导护栏；tests/test_dual_robot_profile_launch.py仅移除旧dual/single launch绑定的两项，保留profile/namespace/helper契约；tests/test_p1_moveit_config.py拆旧single launch断言，保留正式模型/config验证。旧probe/smoke的launch读取与B6联动。

构建：CMakeLists.txt移除fake PROGRAMS安装项，保留aggregator与所有sampler项；检查目录安装后不再含旧launch。必须空build/install/log clean colcon + C++ gtest，不使用源树install。

### B5：validate/demo公开旧入口退役（需要明确确认）

REMOVE：

	run/demo.py、main.py、validate_plan.py、smoke_scene_v2.py、audit_piper_usd.py
	src/rocobrick/cli.py

MIGRATE pyproject.toml删除rocobrick-validate project.scripts，仅保留piper-run；README、AGENTS和config说明改成当前产品接口，保留并整合原有config README dirty功能说明。CLI专属测试按实际消费者展开；共享contracts/STAP/TCP/binding验证不随CLI删除。run/user_config.json先核对是否唯一正式输入，不按目录整删。

### B6：维护工具退出生产树（建议迁仓外工具集）

MIGRATE到独立维护归档/工具集（不只是换到生产仓库另一个目录），同时转移专属工具测试与用法：

	tools/*probe*.py、smoke*.sh、dds_type_matrix*.py、dds/
	tools/p1_fake_joint_states.py
	tools/sim_collision_hulls.py、task_c_support_station_search.py
	tools/publish_piper_l_tcp_calibration.py、validate_piper_assets.py
	isaac/calibrate_piper_l_tcp.py

此为候选分组，执行前展开精确路径和消费者，不能用通配符直接删除。KEEP运行时加载的hulls JSON、clearance门禁及校准产物。

联动：compose.yaml仍挂载tools/container_dds_probe.py，须移除旧诊断挂载/说明或与Compose诊断入口整体迁移；不可让Compose引用不存在文件。docker/ros.Dockerfile和ros-entrypoint.sh默认KEEP作为正式镜像构建链。

测试重点：tests/test_clearance_gate.py只迁生成工具/可复现性测试，GJK/FK/held-brick/sampling门禁保留；tests/test_piper_l_tcp_calibration.py区分工具过程和当前校准契约；DDS multiprocess报告当前仍属正式前置，保留验证语义，不通过删除报告测试绕过生产gate。旧p1/r2 probe专属测试随工具迁出，混合的MoveIt/runtime/StrEnum安全断言保留。

### B7：Python依赖裁剪（独立后续批次）

候选：grpcio/grpcio-tools、openpi、lerobot、isaaclab、chex、torchvision、torchaudio、huggingface_hub以及torch。先查BrickSim/Isaac传递依赖及运行导入；只在独立环境编辑pyproject、重算锁并验证，禁止直接改当前.venv或覆盖现dirty uv.lock。无直接import不是卸包证明。当前批次默认不改锁。

## 3. runs分批策略：单独授权，不混入代码裁剪

R0 MIGRATE：先完成B0以及归档索引；保留正式DDS prerequisite、模型/TCP/calibration artifact，不按“旧报告”整组删artifacts。

R1 REMOVE候选：历史录屏PNG frames。优先报告已分析的最大8个目录内精确帧路径、字节数、当前是否写入、保留MP4/关键帧/证据方案，确认后仅删清单帧；不因EXECUTION_FAILED就删整次运行。

R2 MIGRATE/REMOVE：INVALID_INPUT/重复预检按输入hash+原因+代码版本选代表；成功、安全失败、关键新缺陷单独归档。每例保留原始输入/哈希、argv/env、代码dirty快照、result/preflight/provenance/events、operation-plan-run、cancel/terminal/cleanup与关键日志。缺失材料必须标明，不能宣称完整可复现。

R3：251个tracked历史文件移出index与磁盘删除分开列清单。未经专门确认不执行git rm、批量物理删除或提交。最终生产树不携带历史runs，仅规定新run输出目录。

本轮不重新统计26.88GiB；上一轮PNG约24.63GiB/91.6%只是优先级依据，不是本次释放空间承诺。

## 4. 隔离验证与回应用门禁

1. 获准后在仓库外创建新baseline与candidate（包含当前dirty与必需untracked活文件），不从HEAD clean checkout丢失功能修改。保存逐文件SHA、tracked/index diff和untracked清单；vendor按审计要求保留Git元数据。旧/tmp副本仅参考，不作持久备份。
2. baseline/candidate不带历史runs整树、源树build/install/log/缓存；B0前必要历史原字节放明确受控输入/归档位置。分批候选删除都有精确manifest、被删除测试节点及理由，保留能力覆盖不能下降而不解释。
3. 首轮可使用原Python环境跑快速离线对照，但必须记录sys.path/module.__file__，所有第一方导入指向candidate/src，不能从原editable源码回退；这不算fresh install验收。
4. candidate独立Python环境+锁定依赖做安装/入口/跨进程模块验证；未做依赖批次前不改原锁与原.venv。验证piper-run帮助/无运动输入拒绝、ROS worker模块和host启动文件；不得为import测试无授权启动Isaac。
5. clean ROS：在正式Humble镜像内只挂candidate源码，使用空build/install/log，不source原install，排除旧AMENT/CMAKE/PYTHONPATH overlay；colcon构建agx_arm_description和rocobrick_piper_l_moveit_config。检查安装launch/helper/xacro/校准hash/controller、sampler.so/plugin XML与C++ gtest。原启动脚本有“setup/.so已存在即不构建”逻辑，必须消除旧产物遮掩。
6. 每批focused tests，再跑保留能力全集及Ruff对照。历史2557通过/27失败/2skip、2679 lint是已知基线，不是本轮结果。失败按保留节点逐项比对，不用删除HANDOVER/mimic安全测试制造全绿。
7. 静态/离线/clean构建不等于真实启动；需另获真实运行许可后，用唯一新run路径验收所选技能、失败拒绝、cancel/terminal/cleanup，不覆盖旧输入，不触碰他人资源。Task C覆盖不代表全技能覆盖。
8. candidate验证通过后给出精确diff、测试节点变化、失败对照与回应用清单；原树重新核对preimage哈希。若用户又有新修改立即停止冲突项，逐块合并，不reset/clean。每批原树回应用须在授权范围内；不提交。

## 5. 推荐确认方式

建议先确认：
- 产品范围保留正式operation-plan全技能，退役旧gRPC/pregrasp/phase/独立旧抽象、旧single/launch以及validate/demo比赛入口。
- 先批准B0–B4在隔离副本实施并测试；B5公开入口与B6维护工具去向可以一并确认或后续单批确认。
- 当前不批准任何runs物理批量删除、依赖卸载/改锁、真实机器人/Isaac启动或Git提交；R1/R2/R3分别提交精确清单再确认。
- 代码只在隔离副本先试；原树回应用在看到验证结果与diff后再确认。

未确认前停在本清单。维护工具默认建议迁仓外可复现工具集，而不是为了保留旧测试继续污染生产树。
