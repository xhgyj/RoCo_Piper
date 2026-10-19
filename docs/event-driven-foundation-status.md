# 事件驱动重构：第一批离线基础实施状态

本批仅实现离线契约、依赖准入模型及架构检查。**生产入口仍是
`piper-run` + `robot-operation-plan/v1`；没有接入 v2，也没有开放新运动并发。**
没有启动 ROS、Isaac、MoveIt、Docker 或发送运动。下列代码不能被当作运行时安全认证。

## 已落地的边界

| 路径 | 职责与范围 |
| --- | --- |
| `src/rocobrick/task_plan/schema.py` | 不可变 `TaskPlan`、`TaskOperation`、`TaskDependency`；保留上层任务分配和数组原顺序 |
| `src/rocobrick/task_plan/parser.py` | `parse_task_plan(document)` 严格解析 `robot-task-plan/v2`；拒绝旧层字段和未适配的 HANDOVER |
| `src/rocobrick/task_plan/dependency_graph.py` | 显式边、环/重复/未知身份检查；确定性的前驱、后继与 ready 查询 |
| `src/rocobrick/contracts/` | 阶段身份、物理证据、阶段目标/候选/约束引用、接触、权限、运动承诺和分类失败 DTO |
| `src/rocobrick/scheduling/scheduler.py` | 无运动副作用的事件准入内核；`admit` 只表示可请求协调，绝非执行授权 |
| `src/rocobrick/scheduling/resource_manager.py` | 原子任务资源租约；复用阶段资源契约，不管理轨迹预约或控制器通道 |
| `tests/architecture/` | 新核心的静态导入方向、无循环、无外部框架依赖检查；不声称旧技能已完成解耦 |

v2 形状以设计文档第 3 节示例为基础，只有显式 `physical_success` 依赖。
PICK、PLACE_DOWN、PLACE_UP、SUPPORT_TOP、WAIT 可以表示；HANDOVER 明确拒绝。
实体与目标是待绑定的物理身份引用，不代表已经验证目标存在或几何可行。
可选 `parameters` 仅作深度冻结的 JSON 数据保存；技能参数解释及合法性验证尚未迁移，
不能用它绕过正式目标、接触或运动权限字段。

## 准入模型的安全语义

- `WAITING → ADMITTED → SUCCEEDED` 表示申请/验证状态，不是控制器状态。
- 外部依赖只由整个 operation 的独立物理完成验证满足；内部阶段完成不能代替。
- `VerificationPort` 是待实现的可信验证端口。测试替身不是生产物理验证器。
  它必须绑定会话，检查身份、attempt、新鲜度、场景/模型来源及完整物理和终态证据。
- 成功不自动释放租约；释放还必须证明无持续 holding/support/contact 义务。
  因而真实 PICK→PLACE 不能使用测试中的“无持续义务”释放路径，必须待实现显式移交。
- 失败阻塞真实后继；未知终态先保守隔离整个计划。安全释放已失败操作后，才允许
  无关联操作继续。取消请求、失联和事实撤销绝不等于停止。
- 原会话/原 attempt 之外的事件不能完成当前操作；相同事件重复投递幂等；
  相同 event ID 配不同内容拒绝。撤销会将已准入后继标为不确定并保留资源。
- 原子准入仅适用于一个事件循环，无跨线程/跨进程事务保证；没有重启恢复能力。

## 本批有意未做

1. 生产 CLI 切换、结构绑定、不可变输入捕获对 v2 的集成。
2. 唯一执行入口及 prepare/start/cancel 迁移、owner/epoch 运行时检查和幂等发送。
3. 技能阶段展开、技能完成谓词注册、阶段图调度和有物理连续性证明的租约移交。
4. 世界事实注册表、信任验证器注册表、观测新鲜度与依赖版本失效。
5. 在途轨迹预约、连续碰撞/停止包络、时间窗、避让和联合重规划。
6. 后继目标绑定、可抓可放候选求解、实测抓持复核以及后端替换。

特别注意：DTO 的 `from_dict` 仅验证结构，不验证声明是否真实；证据 checks 为 true
不自动证明 holding；MotionCommit 字段齐备不自动获得运动权。候选、约束及接触条件
引用尚无注册表解析；不得直接把这些离线 DTO 接到旧运动发送接口。
阶段完成/中断策略、模型来源细分、场景依赖版本、切换点、停止策略、验收凭证等
完整生产契约还需在相应垂直切片补齐，不能宣称第一批已冻结全部运行时协议。

## 验证方法

仅运行离线 pytest 与 Ruff；使用锁定环境，不安装依赖：

```sh
PYTHONPATH="$PWD" ~/.local/bin/uv run --no-sync python -m pytest -q -p no:cacheprovider \
  tests/task_plan tests/contracts tests/scheduling tests/architecture
~/.local/bin/uv run --no-sync ruff check \
  src/rocobrick/task_plan src/rocobrick/contracts src/rocobrick/scheduling \
  tests/task_plan tests/contracts tests/scheduling tests/architecture
```

保留旧生产链回归：`tests/test_robot_operation_plan_and_scheduler.py`、
`tests/test_multi_robot_scheduler_contract.py`、`tests/test_successor_pre_phase_scheduler.py`、
`tests/test_coordinated_execution.py`、`tests/test_pick_candidate_selection.py`、
`tests/test_piper_run.py`。A→C、B 独立测试证明的是 C 的申请资格不依赖 B，
不是证明两个机械臂实际执行区间可安全重叠。

下一垂直切片应先提取执行所有权和唯一发送入口（保持串行），再迁移一个完整技能
阶段链；旧路径迁移前不得让新旧执行器同时控制同一会话。

## 本次验证结果

- 最终集成：上述新核心及六个生产回归文件合计 **305 passed**；新核心单独为
  **152 passed**。修改范围 Ruff check、Ruff format 检查及 git diff --check 通过。
- 子代理曾在实施中间状态执行全量离线 `pytest tests`：
  **28 failed, 2467 passed, 2 skipped, 1 warning**。没有变更前同命令的基线，
  不能把这些失败断言为预存问题，也不能称全仓测试全绿。最终状态未重跑全量。
- 失败集中于 `tests/test_handover_preparation.py`（18 项）、
  `tests/test_handover_program.py`、`tests/test_handover_runtime_safety.py`、
  `tests/test_handover_scene_planner.py`、`tests/test_handover_staged_execution.py`、
  `tests/test_host_bridge_scene_state.py` 和 `tests/test_ros_moveit_runtime_contract.py`。
  代表错误包括 `LOCAL_ALIGNMENT_DISTANCE_EXCEEDED`、
  `NO_FEASIBLE_HANDOVER_IN_PREPARATION_REGION`、
  `TypeError: box_geometry_digest() takes 0 positional arguments but 3 were given`
  以及 mimic violation 的 opening/status 断言。未修改这些模块来使测试变绿。
- 检查到的运行时契约测试使用注入 fake；host bridge 测试用 stub，不启动 Kit/rclpy。
  pytest 的显式子进程调用用途为离线 CLI preflight、导入检查、DDS 配置生成及可选
  xacro 展开，不是仿真或运动启动命令。后续子代理应只运行明确列出的离线测试范围，
  不自行扩展为全仓测试，以保持验证边界和结果记录可审计。
