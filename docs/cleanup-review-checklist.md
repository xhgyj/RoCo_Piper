# 大清理评审清单

用户已确认本阶段边界：保留已验证候选，先评审清理清单；不提交、不删除数据。此清单不授权后续移动、删除、移出 Git index 或提交。

## 已交付的初步基线

- 认知、启动链路、质量基线：docs/repository-baseline-20261015.md。
- 依赖版本和恢复缺口：docs/dependency-recovery-baseline.md。
- 仓库外隔离候选：/tmp/roco-clean-baseline-20261015-7kipxnd1/candidate；约 778 MiB 文件内容（du 约 783M）。未含历史运行输出、ROS build/install/log、.venv、.runtime、Git 元数据、代理状态、缓存或 .orig；仅保留 3 个必要历史输入例外。
- 全量测试：2557 passed / 27 failed / 2 skipped，与原工作区一致；使用原 .venv，候选 src 优先。不是 clean Git checkout、全新依赖安装或端到端成功证明。
- 候选清单：/tmp/roco-clean-baseline-20261015-7kipxnd1/inventory.json，共 1643 项（含 symlink 记录），SHA-256 a712a880e5d82cb97eba869b2daa111d572594aa7d40ade3d9a9408f939aba78。这是隔离候选当时的内容快照；后来工作区文档/ignore 修改不自动同步。/tmp 非持久归档，不应作为唯一备份。

## 分组评审

| 分组 | 当前处理 | 后续需确认 |
| --- | --- | --- |
| 原有 46 个 tracked 修改/删除 | 保留；未回滚，未提交 | 功能作者确认哪些属于候选基线；不要把所有 diff 混进“清理提交” |
| clearance 源码、hulls、4个新增测试 | 可见、保留 | 归属及测试覆盖；需纳入未来正式基线，不得当垃圾忽略 |
| probe/smoke 与 DDS 工具 | 可见、保留 | 区分仍被测试加载的工具与可淘汰实验工具；必须先改引用并测试 |
| camera xacro/DAE | 保留 | 明确模型来源、许可及未来跟踪方式 |
| third_party 声明与两 vendor | 保留；核验 HEAD/hash | 固定恢复流程、初始化 sim URDF 子模块；资产许可仍待确认，不随意复制发布 |
| robot_assets 与 planner_ws | 保留 | 活跃配置、校准、Docker 仍引用；退役旧链路需要独立验证 |
| artifacts/model-audit / calibration / scene | 保留 | 固定输入和证据来源；不可混同临时日志 |
| artifacts/phase* | 不做全目录 ignore | 分离测试读取的冻结 JSON、可归档日志及可重新生成的报告；记录哈希后迁移 |
| runs 中 3 个冻结输入 | ignore 显式例外，可见且保留 | 未来完成分段路径与 artifact provenance 的整体迁移后再取消兼容路径 |
| 251 个 tracked runs 证据 | 保持跟踪，未操作 index | 明确保留证据与仓库外归档位置、恢复清单后，评审是否移出 index；ignore 不等于取消跟踪 |
| 其余 runs（约28G） | 本地原样保留，新增产物忽略 | 选择历史成功/重要失败证据，校验归档，再单独授权物理清理 |
| ROS build/install/log 与缓存 | 忽略，未物理删除 | 验证可重建后是否释放空间；安装树链接环另查 |
| scene_snapshot_collector.py.orig | 原样保留 | 与当前文件和已有 Git 版本比较后再确认是否归档/删除 |
| uv.lock 原有修改 | 原样保留 | 判断变动来源及锁定依赖可恢复性；本轮未 uv sync |

## 后续验收门槛

1. 源码/工具/输入/依赖/历史证据分别成组确认，保留现有作者工作。
2. 评审清单批准后才执行归档和路径迁移；源字节、SHA-256 与恢复信息完整。
3. 在正式候选 Git 检出中验证依赖恢复与离线套件；不把当前已知 27 失败藏成 skip，也不放宽安全契约。
4. 真实仿真验收另用唯一新 run 目录；不能把旧视频或历史成功当当前 Task C 成功。
5. 正式基线提交和删除均需另行确认范围。

本阶段完成的是项目初步认识、低风险噪音隔离、可复测候选和待评审清单；不是完成全部仓库瘦身或修复全部功能。
