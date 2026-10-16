# B0–B4 原仓库回应用完成

已将确认过的候选变更回应用到 /home/mzictt/RoCo_Piper；未处理HANDOVER功能缺口。

## 实际变更与保护

- 精确manifest 157项：114删除、36修改、7新增；全部迁移后SHA匹配candidate。
- 持久备份：/home/mzictt/RoCo_Piper_snapshots/cleanup-b0-b4-before-apply-20261016T023345Z。
- 备份包含186个迁移前/原dirty相关文件、原tracked/index补丁、preimage索引、候选diff/manifest、独立审计和迁移后测试证据。
- 150个manifest preimage均与备份SHA匹配；7个新增项原先不存在。写入前逐项校验，先写新内容后删除旧文件，逐路径记录执行日志；未进行reset/clean。
- 清单外1470个inventory文件均与baseline字节一致。原dirty复杂文件的6个功能节点AST保持，HANDOVER测试保护尾部逐字节一致。
- 原uv.lock、pyproject、skill_params、HANDOVER生产实现未变。原runs三份历史计划仍在且哈希一致，251个tracked历史runs文件未移出index/删除。
- 未复制候选缓存、ROS build/install/log到原树。旧ROS install可能仍存在于原树；本次构建验收使用隔离空目录，未借它掩盖缺失源码。
- 原树新增4个analysis交付证据文件，独立审计确认不属于功能源码变动。
- HEAD和分支未变，无暂存或提交。当前tracked状态71修改、115删除（含原有删除），共186项；另有原先与本次新增的untracked，不能把它们当垃圾清除。

## 原树复测

DeepSeek独立运行原树全量tests：2441 passed / 28 failed / 2 skipped。与candidate 2471节点逐项完全一致：0缺失、0新增、0状态变化、0新失败。

失败构成仍为原baseline 27项全部保留，加已在未裁剪baseline复现的HANDOVER attachment digest调用TypeError。按用户要求不修复、不skip/xfail。

Ruff原树2236诊断，与candidate按相对路径/code/message计数完全一致；不宣称全量lint通过。命令写入了原有gitignored Ruff缓存，但未改源文件。

正式multi session/ROS worker模块导入和piper-run help通过；无自有运行容器或pytest/Isaac/ROS残留。无真实ROS节点、Isaac或运动启动。

## 独立核验

本轮两项明确任务委派DeepSeek：逐文件postimage/备份/dirty/runs核验，以及原树全量测试与lint逐项对照。主代理负责迁移范围、逐路径冲突保护和原树修改。

持久证据：
- candidate-change-manifest.json、candidate-source.diff、preimage-index.json、applied-paths.json、apply-result.json
- postapply-deepseek-audit.json、dirty-node-preservation.json
- applied-tests.xml、applied-tests.log、applied-ruff.json
- applied-vs-candidate-node-comparison.json、applied-vs-candidate-ruff-comparison.json

## 后续边界

仅B0–B4已迁移完成。B5旧validate/demo公开入口、B6维护工具迁出、B7依赖裁剪及历史runs物理清理仍未执行；部分旧诊断工具仍引用已退役launch，需要后续成组处理。正式全技能分支、输入审计和取消/terminal/cleanup门禁保留。
