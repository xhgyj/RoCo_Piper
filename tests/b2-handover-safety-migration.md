# B2 handover安全测试迁移节点映射

范围：仅candidate/tests/test_handoff_contract.py、test_handover_runtime_safety.py及本报告。未改生产源，未删skills/handover.py，未改原树或baseline。真实阅读direct_handover、handover_live、handover_contact、handover_observation、handover_geometry、coordinated_handover及既有coordinated生命周期/geometry/finger/joint planning测试。tests中不存在test_direct*.py，也未发现Direct backend原测试，新增真实Direct backend拒绝路径覆盖。

## 结果

迁移前这两文件+coordinated lifecycle：58 passed。
迁移后六文件：148 passed、1 failed（1.94s）；两独占文件最近单测为43 passed、1 failed（新增foreign identity四节点后）。使用candidate/src优先原/home/mzictt/RoCo_Piper/.venv，无ROS/Isaac启动。direct_handover.__file__已核对来自candidate/src。

唯一失败：test_transfer_attachment_uses_post_approach_receiver_tcp，进入真实LiveSimulationHandover._transfer在src/rocobrick/adapters/moveit/handover_live.py:849失败：box_geometry_digest(frame,lo,hi)调用keyword-only函数，TypeError。这是正式生产路径缺口，原helper测试没触达它。保持为普通失败测试，不skip/xfail、不patch生产函数、不改src。失败前尚未dispatch；其attachment/digest后续断言保留，无法宣称已验证成功。foreign session/manifest两拒绝节点已通过，证明无dispatch。

语义差异：DirectHandoverBackend异常catch固定terminal=False，即使没有dispatch；新增test_direct_backend_pre_dispatch_failure_is_fail_closed_not_terminal_safe验证现保守拒绝，不冒称旧terminal-safe等价。coordinated backend的no-dispatch terminal-safe契约另由迁移后同名节点保留。Direct完整成功/continuation未新增覆盖，不声称formal lifecycle成功等于Direct end-to-end。

## 每旧节点映射

### test_handoff_contract.py

- test_pre_phase_evidence_requires_every_lifecycle_gate -> 原节点原语义保留。
- test_wrong_bindings_never_match（5参数） -> 原节点保留。
- test_shared_snapshot_rejects_staleness_and_unknown_physical_state -> 原节点保留。
- test_handover_contract_requires_distinct_robots_and_real_booleans -> 原节点保留。
- test_handover_follow_sample_count_is_strict（5参数） -> 原节点保留。
- test_handover_commit_atomically_moves_holding_owner -> 同名节点改真实ledger.transfer_holding，确认donor撤销/receiver确认/owner移动，不用旧Handler.commit。
- test_handover_partial_evidence_preserves_donor_owner -> 同名节点改ledger.transfer_holding，缺release门禁拒绝且owner和donor fact保持。
- test_handover_unknown_state_blocks_both_robots_until_reconciled -> 原节点保留。
- test_handover_operation_backend_adapts_and_commits_scheduler_result -> 同名节点改真实CoordinatedHandoverBackend，复用已有synthetic physics/action fixture，确认最终follow、owner提交，以及donor release发生时双方fact已撤销。
- 新增test_handover_evidence_foreign_identity_cannot_transfer_owner（operation/donor/receiver/object四参数） -> ledger拒绝foreign evidence且不转移owner。

HandoverRequest/Evidence helpers保留，它们属于正式strict contract，不含旧skills.handover依赖。

### test_handover_runtime_safety.py

- test_receiver_target_is_measured_object_boundary_in_its_own_base（2 robots） -> 同名节点用正式calibrated_target+HandoverPlanningRequest上下文；保留旋转/平移base和frame/tip/provenance/tolerance断言。旧side-grasp固定position 0.28/0.33/0.1135及旧precontact orientation不再是正式paired准备几何契约，明确退役其数值。
- test_receiver_contact_boundary_keeps_the_pick_floor_tip_clearance -> 同名节点改真实IsolatedHandoverGeometry：初始无碰撞，升高捕获floor后拒绝。退役旧PICK z clamp数值，不退役floor安全含义。
- test_handover_contract_rejects_default_or_foreign_receiver_frame（三参数） -> 原strict request拒绝节点保留。
- test_distant_stationary_object_cannot_prove_receiver_contact（三轴） -> 同名节点用正式parse_handover_observation/verify_co_hold，stationary+plausible opening但缺真实opposed pad contacts必拒绝。
- test_cohold_geometry_failure_never_transfers_scene_or_releases_donor -> 同名节点真实coordinated backend缺receiver contact，不允许donor_open/scene_transfer；保护owner、不发布receiver holding。
- test_handover_no_dispatch_failure_does_not_claim_motion -> 同名节点真实coordinated backend foreign executor，terminal-safe且无motion/transport调用，giver fact仍在。
- test_attempted_transfer_failure_invalidates_confirmed_holder（scene-transfer/release） -> 同名节点参数改正式scene_transfer/donor_open阶段，真实coordinated phase失败后取消、unsafe lease、双方fact撤销和uncertain_handover；旧scene-before-release顺序不强加新正式八阶段流程。既有test_every_phase_failure_stops_both_and_retains_uncertain_custody（PHASES全阶段）也覆盖。
- test_transfer_attachment_uses_post_approach_receiver_tcp -> 同名节点直接真实LiveSimulationHandover._transfer，以当前sample计算relative bounds及digest；现在暴露上述生产TypeError，保留失败。原foreign provenance子断言拆成test_transfer_attachment_rejects_foreign_provenance_before_dispatch（session/manifest），两节点通过。
- test_failure_before_executor_has_explicit_no_dispatch_proof（ik/plan/readback） -> 原节点原字节保留。
- test_exception_or_invalid_evidence_after_executor_is_unsafe（2参数） -> 原节点原字节保留。
- test_worker_planning_failure_blocks_admission_but_allows_orderly_stop -> 原节点原字节保留。

从def _planning_fixture():到文件尾与baseline逐字符/UTF-8字节一致，含原dirty +7行unified_robot_description假体及全部三个正式测试；仅上游imports重组。两文件无skills.handover import/helper使用。

## 原有正式证据并测

- test_coordinated_handover_lifecycle.py：complete transfer、every-phase failure、missing physical proof、pre-dispatch、cancel lease、profile qualification、mutation、missing giver contact；全部通过。
- test_handover_joint_planning.py：strict profile/identity/provenance、actual contact/follow和planning gates；全部通过。
- test_handover_finger_contact.py：正式FingerContactGuard phase/身份/历史连续/force/region guard；全部通过。
- test_handover_geometry.py：native full geometry、floor/world collision、collision exemption/asset provenance/limits；全部通过。

本轮清理不能发布“全绿”或“正式transfer已验证”；上游协调者应独立决定生产修复、剩余风险接受与旧模块物理删除。
