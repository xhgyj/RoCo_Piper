# RoCo_Piper B0–B4 隔离裁剪结果（等待原树回应用确认）

## 结论

B0–B4 已在新隔离候选完成代码/测试裁剪与当前可执行的离线和构建验证。原树未回应用；不是全绿，也不是全技能实机/仿真运行认证。新增真实HANDOVER回归揭示既有生产缺口，建议先单独审阅再决定回应用。

## 位置与完整性

- 原树：/home/mzictt/RoCo_Piper，HEAD 14f1c4b，分支rebuild/piper-isaac-moveit。
- 新工作根：/tmp/roco-live-cleanup-20261015-session2。
- baseline与candidate分别保存当前dirty基线和裁剪结果。baseline按当前工作树复制，不是clean HEAD；包含必需untracked模型/clearance/vendor，排除原Git、.venv、build/install/log及历史runs整树。
- 原tracked差异SHA-256始终为be575cafd85c3675f3c5d0613a4ad880bf69c59d792898a8b2163f240b5e9539；51项tracked dirty；index为空。
- 没有reset/clean、提交、原runs删除、改锁/卸载原环境或真实ROS/Isaac启动。只运行无节点容器构建/导入/测试，最终docker ps为空。
- /tmp不是持久备份；原树回应用前需保存交付清单、diff和必要归档到持久位置。

## 已执行范围

1. B0：两份测试剩余runs计划读取迁fixture；三计划重新校验原字节/哈希。TCP测试用显式历史路径索引解析SHA寻址.bin审计对象，未知路径失败。归档pregrasp、计划、events和approach request原字节，不改TCP artifact；另有仓外provenance-archive副本。候选内三份历史计划已精确删除（原树不动），focused133通过。
2. B1：退役planner_ws/proto、planner Docker入口、gRPC generated/client/contract、pregrasp/playback和专属测试；混合资产/StrEnum/许可/禁导测试保留和迁移。
3. B2：退役旧phase backend、stap_schedule/supervision/verification及专属测试；skill_execution消费者迁真实包。旧HandoverHandler删除，混合安全断言迁ledger/coordinated/observation/live API；严格契约保留。_planning_fixture起至文件尾与baseline逐字节一致，包含原dirty+7行功能测试。
4. B3：消费者迁到classified owners后删除6个转发模块；同名execution.py/skills.py遮蔽文件、未用waypoints/空scheduling移除。根包和skills包去隐式旧API导出；不把其原来拉入的contracts/scheduler/task_facts直接当死代码。collector.orig比较只缺当前新增manifest/host验证，无独有新增行为，原字节归档后从候选删除。
5. B4：精准删除旧单臂session分支及3旧launch/fake controller；保留multi共享runner/status/error/constants/worker链。CMake仅删fake安装项；single测试31节点和旧launch两节点退役，其他共享契约保留/迁正式launch。
6. 同步候选README、AGENTS、third_party README、正式launch说明与相关文档测试。README保留上游历史上下文但明确不作当前部署承诺。

## 变更规模

以baseline对candidate源文件对照（排除Git/缓存/runs临时输出）：157路径变化：114删除、36迁移修改、7新增。删除文件逻辑字节36,394,383（约34.7MiB），主要旧planner资产/代码；不是原树释放空间。另单列候选3份历史计划移除，原runs完全不动。候选测试产生的新run输出保留在工作副本用于诊断，不算交付源变化。

精确权威清单：candidate-change-manifest.json（逐路径before/after SHA），candidate-source.diff（UTF-8源码差异；二进制资产及.bin以manifest哈希核验），各B批removal/test-retirement记录作过程证据，最终manifest哈希优先。

## 实测验收

| 验证 | baseline | candidate / 结果 |
|---|---|---|
| Python全量测试 | 2557 passed / 27 failed / 2 skipped | 2441 passed / 28 failed / 2 skipped |
| 基线失败节点对照 | 27 | 27全部保留，无消失 |
| 测试节点集合 | 2586 | 2471；131旧节点退出、16新节点进入（含重命名/参数身份变化，不等同全是新增功能测试） |
| Ruff | 2679诊断 | 2236诊断；按路径+code+message计数，无新增签名诊断 |
| clean colcon build | 不借原install | 2包成功：agx_arm_description与rocobrick_piper_l_moveit_config |
| C++测试 | 新build/install/log | 11 tests，0 errors / 0 failures / 0 skipped |
| 安装产物无节点验证 | 新install，仅source Humble+新overlay | 10/10检查通过 |

无节点安装验证实际覆盖：安装版launch/helper与aggregator可导入、真实xacro展开、校准artifact哈希、双臂config生成、namespaced controller/topic、sampler ldd无缺依赖+dlopen+动态符号/XML接线、16安装文件与candidate源码逐字节一致。未验证运行中的MoveGroup/plugin allocator实例化、DDS/readiness、动作、cancel或cleanup；不宣称运行成功。

独立Python验证：新package-venv非editable安装candidate（--no-deps）并单独装已锁numpy1.26.4；-I导入piper_run/ros_session_entry/worker来自独立site-packages，旧planner_client/skills.handover不存在；piper-run --help成功。它只验证打包/导入边界，不是完整uv sync --locked/fresh Isaac依赖环境，不冒充完整部署验证。全量测试使用原.venv但candidate/src优先，未修改原环境；跨进程测试亦读取candidate。

## 必须显式保留的风险

### 新增回归暴露既有HANDOVER缺口（未修复）

测试test_handover_runtime_safety.py::test_transfer_attachment_uses_post_approach_receiver_tcp进入正式LiveSimulationHandover._transfer，handover_live.py:849调用box_geometry_digest(frame,lo,hi)，但函数是keyword-only(object_id,frame_id,minimum_m,maximum_m)，触发TypeError。

该回归在未裁剪baseline上同样失败，且对应生产文件未被裁剪修改。测试保留普通失败，不skip/xfail，不在本清理批修生产。成功attachment后续断言尚未通过，不能声称迁移安全证据全等价。

另：Direct backend pre-dispatch异常现有terminal=False，与旧terminal-safe语义不等价；保留现fail-closed测试及差异说明，不悄悄改语义。Direct完整成功/continuation未做端到端覆盖。

### 当前范围不是最终全仓极简完成

B5 validate/demo公开入口、B6维护工具迁仓外、B7依赖改锁与原runs清理均未执行。旧维护probe/smoke仍含退役launch命令，README已标不支持的历史诊断接口；需后续整组迁出/退役，不能当现部署指令。正式全技能分支和运行安全门禁均保留。

## 原树回应用建议

1. 先决定是否单列修复HANDOVER既有digest调用缺口；结构清理与功能修复分开，不以全绿掩盖缺口。
2. 用户确认回应用后，重新检查逐路径preimage SHA及tracked/index差异；任何用户新改动冲突停止对应项、逐块合并，禁止reset/clean。
3. 只按最终源manifest应用B0–B4，不复制候选runs、缓存、build/install/log，不触碰原锁/.venv。runs原始输入及历史大树仍需专门归档/删除授权；候选中的3份输入移除不自动等于原runs删除授权。
4. 独立保存归档对象/索引、候选diff、测试节点对照及构建证据；不提交。
5. B5/B6和runs按前清单另批确认；真实启动/cancel/terminal/cleanup仍须另获许可并使用唯一新run路径。

## 主要证据文件（位于工作根）

- inventory.json、tracked.patch、index.patch、status.txt
- candidate-change-manifest.json、candidate-source.diff
- test-node-comparison.json、baseline-tests.xml、candidate-tests.xml、baseline-tests.log、candidate-tests.log
- baseline-new-regression.log（新回归在未裁剪基线复现）
- baseline-ruff.json、candidate-ruff.json、ruff-comparison.json
- B1-B3-removals.json、B3-forwarder-removals.json、B2-handler-removal.json、B4-test-retirement.json、candidate-runs-removal.json
- ros-clean-build.log、ros-clean-build/、ros-clean-install/、ros-test-log/、ros-installed-validation.json
- candidate/tests/b2-handover-safety-migration.md（逐旧安全节点映射）
- provenance-archive/（仓外审计原字节与collector备份）
