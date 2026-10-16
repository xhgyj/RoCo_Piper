# third_party — 第三方上游锁定说明（P0.1）

本目录锁定 RoCo_Piper 依赖的第三方上游仓库的精确版本与许可事实。

## 唯一版本源

`manifest.lock.yaml` 是本仓库所有第三方上游代码/资产的**唯一版本源**。
任何关于"我们用的是哪个版本、哪个 commit、许可文件及其哈希"的问题，一律以
`manifest.lock.yaml` 为准；其它任何目录下的副本（包括历史遗留 vendor 目录）
都**不是**版本依据。

## 禁止浮动 branch clone

**禁止以浮动 branch 方式 clone / 跟踪上游**（例如 `pip install git+...@ros2`、
不锁 commit 的 `git clone -b ros2` 后直接构建）。上游分支随时可能前移，浮动
引用会破坏可复现性。构建入口（包括所有 Dockerfile 与镜像构建脚本）同样
**禁止 clone 浮动 branch**，只允许按 manifest 锁定的 commit 导入。所有导入必须：

1. 只 checkout `manifest.lock.yaml` 中记录的 **exact commit**；
2. 在 checkout 后核对 `git rev-parse HEAD` 与 manifest 中 `commit` 完全一致；
3. 校验许可源文件 sha256 与 manifest 中 `license.sha256` 完全一致
   （`sha256sum <source_path>`，对锁定 commit 的 raw 内容计算）；
4. 若上游含子模块（gitlink），checkout 后必须
   `git submodule update --init --recursive`，并核对子模块
   `git rev-parse HEAD` 与锁定 commit 树中的 gitlink 完全一致
   （agx_arm_ros 的 gitlink 见下文专节）。

## 当前锁定条目

| 仓库 | branch | commit | 许可 | source_path | 导入目的 |
| --- | --- | --- | --- | --- | --- |
| agx_arm_ros | ros2 | `b9ad14de2a6eb818a2d206fbd1adba48343662f7` | MIT | `LICENSE` | Piper ROS 2 Humble 驱动（description / moveit / ctrl），导入到 `ros_ws/src/agx_arm_ros` |
| agx_arm_sim | master | `f8cd8b147c75d59e14f90fb0646770eefa268ed0` | MIT（仅 README 声明） | `README.md` | Piper / Piper L 仿真资产（URDF/Xacro + USD），导入到 `isaac/vendor/agx_arm_sim` |

## agx_arm_ros 子模块 gitlink（导入必须递归初始化）

锁定 commit `b9ad14de2a6eb818a2d206fbd1adba48343662f7` 的树中，
`src/agx_arm_description/agx_arm_urdf` 是一个子模块 gitlink：

- gitlink commit：`f6642ce0d7872c686f29c99e9e10cd23d1d49313`
- 子模块 URL：`https://github.com/agilexrobotics/agx_arm_urdf.git`
  （`.gitmodules` 声明 branch: main，但子模块必须停在 gitlink commit 的
  detached HEAD，不得跟随浮动 branch）
- 只读验证：`git ls-tree b9ad14de2a6eb818a2d206fbd1adba48343662f7 src/agx_arm_description/`
  应对 `src/agx_arm_description/agx_arm_urdf` 输出
  `160000 commit f6642ce0d7872c686f29c99e9e10cd23d1d49313`。

Piper 的 URDF/mesh 与 gripper 关节事实只存在于该子模块内容中：仅 checkout
父仓库时该目录为空，gripper / URDF / mesh 事实无法复验。因此**任何
agx_arm_ros 的导入（含只读核验）都必须在 checkout 后执行
`git submodule update --init --recursive`**，并核对子模块
`git rev-parse HEAD` 与上述 gitlink 完全一致。该 gitlink 是锁定 commit 的
树上对象（随 manifest 锁定的 commit 一起被锁定，本节仅是记录）；升级评审
更换 commit 时必须重新核对并同步更新本节。

## 旧 planner 退役与正式资产

旧 planner_ws/src 链路及其中历史 vendor 拷贝已退役，不是正式运行或版本来源。
robot_assets/piper_l 是正式运行资产，必须与锁定上游逐层核对，不能作为独立版本源。
正式 vendor 保留在 ros_ws/src/agx_arm_ros 与 isaac/vendor/agx_arm_sim；
其 Git 对象仍用于模型审计，不是可清理缓存。唯一版本源仍为 manifest.lock.yaml。

## agx_arm_sim 许可限制与 P0.4 复审

- agx_arm_sim 仓库**没有独立的 LICENSE 文件**，MIT 许可仅来自其 `README.md`
  末尾的 `## License` 段声明（"MIT License"）。
- 因此 manifest 中该条目的 `license.source_path` 为 `README.md`，
  `verified_facts` 含 `no_standalone_license_file: true`。
- **不得把这条许可记录解读为 USD / mesh 资产许可已确认**：仓库内的 USD/网格
  模型可能来自子模块或第三方来源，其许可状态未经审计，
  `assets_license_confirmed: false`，复审安排在 **P0.4**
  （`assets_license_reaudit_phase: P0.4`）。在 P0.4 完成前，不得在交付物中
  声称 sim 资产许可已澄清。
- 该仓库 README 未声明 Isaac Sim 5.1 兼容性
  （`isaac_sim_5_1_compatibility: not_declared`），不得对外宣称 5.1 兼容已获
  上游确认。

## 只读更新流程（git 命令，不创建更新脚本）

本仓库**不创建、也不允许引入自动更新脚本**；第三方上游的核对与升级一律通过
下面的只读 git 命令人工完成。更新第三方前先用只读命令核对上游是否仍与锁定
一致；分支头发生变化时，必须走完整的升级评审（更新 manifest、重算许可哈希、
跑 `tests/test_third_party_manifest.py`），而不是盲目跟随分支：

```bash
# 1) 只读探测：上游分支头是否仍等于锁定 commit（变化则触发升级评审，不自动跟随）
git ls-remote https://github.com/agilexrobotics/agx_arm_ros.git refs/heads/ros2
git ls-remote https://github.com/agilexrobotics/agx_arm_sim.git refs/heads/master

# 2) 在 /tmp 只读拉取锁定版本（不要 clone 进工作区后停在浮动 branch 上）
git clone https://github.com/agilexrobotics/agx_arm_ros.git /tmp/agx_arm_ros
cd /tmp/agx_arm_ros
git checkout b9ad14de2a6eb818a2d206fbd1adba48343662f7
git submodule update --init --recursive  # 必须递归初始化：URDF/mesh/gripper 事实在 agx_arm_urdf 子模块中
git rev-parse HEAD   # 必须等于 manifest 中的 commit
git -C src/agx_arm_description/agx_arm_urdf rev-parse HEAD   # 必须等于 gitlink f6642ce0d7872c686f29c99e9e10cd23d1d49313

git clone https://github.com/agilexrobotics/agx_arm_sim.git /tmp/agx_arm_sim
cd /tmp/agx_arm_sim
git checkout f8cd8b147c75d59e14f90fb0646770eefa268ed0
git submodule update --init --recursive  # 模型资源在子模块中
git rev-parse HEAD   # 必须等于 manifest 中的 commit

# 3) 校验许可源哈希（必须等于 manifest 中的 license.sha256）
sha256sum /tmp/agx_arm_ros/LICENSE
# 期望: 247944c63de1573653722ba784f96123da267105b7b48577e0c8826a24b809f6
sha256sum /tmp/agx_arm_sim/README.md
# 期望: b91918f2972101eee621cba9badbc90e6057b9c5269c0bb7ae00f84195d31399

# 3') 等价的只读核验（用 git show 直接读锁定 commit 的对象内容，不落盘）：
git -C /tmp/agx_arm_ros show b9ad14de2a6eb818a2d206fbd1adba48343662f7:LICENSE | sha256sum
git -C /tmp/agx_arm_sim show f8cd8b147c75d59e14f90fb0646770eefa268ed0:README.md | sha256sum
git -C /tmp/agx_arm_ros ls-tree b9ad14de2a6eb818a2d206fbd1adba48343662f7 src/agx_arm_description/ | grep agx_arm_urdf
# 期望: 160000 commit f6642ce0d7872c686f29c99e9e10cd23d1d49313 src/agx_arm_description/agx_arm_urdf
```

升级流程（需评审后执行）：更新 `manifest.lock.yaml` 的 `commit` / `license.sha256`
/ `verified_facts`，同步更新 `tests/test_third_party_manifest.py` 中的期望值，
并运行 `uv run --frozen python -m pytest tests/test_third_party_manifest.py -q`。
