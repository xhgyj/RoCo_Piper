# 任务 C：技能缺口补齐

范围：当前 scaled2 厚砖；piper_0 执行 PICK / PLACE_UP，piper_1 执行 SUPPORT_TOP。不扩展薄砖、不调整初始化槽位、不改机械臂模型与关节限位。

代码安装位置：远端 `/home/mzictt/RoCo_Piper`，分支 `rebuild/piper-isaac-moveit`。本地 `.remote_stage_task_c` 仅用于编辑与同步，不是执行入口。未提交代码；保留原有未提交改动。

## 本轮实现

- PICK 消费已有计划参数 `successor_skill=PLACE_UP`，选择侧向工具姿态、保持闭合轴横向，让上方 stud 面留空。普通 PICK 仍用原姿态。保留 open → precontact → close → lift；不增加绕路、TRANSIT 或 regrasp。这个新抓取姿态尚未真实验证。
- PICK 的持砖证据补上 attachment、terminal-safe 和后继持有标记；不同机器人执行 SUPPORT_TOP 不会接走携物臂的 robot/part 资源。
- PICK → SUPPORT_TOP 的 successor pre-phase 同时生成携物臂预放置边界和支撑臂上方接近目标，走一个 `dual_arm` combined plan。原始计划不重写；完整真实起始关节状态、持砖变换和碰撞场景进入同一会话。
- 后继 handler 通过独立 `bind_pre_phase(operation_id, receipt)` 接收经过验证的到达凭据，不重复执行公共空间移动。
- SUPPORT_TOP 的工具 Z 指向上砖表面，指尖沿长边布置；TCP 高度考虑实测 5.211 mm 指尖偏移。公共接近之后，使用已有低速 Pilz LIN / guarded TCP 接触执行器；上砖位移、反力超限或反馈失效会停止。支撑确认要求三个推进的同会话 PhysX 顶面接触样本和上砖稳定，不能只凭“砖没动”。不把支撑当成抓取。
- PLACE_UP 移除未实现的 `run_coordinated_pre_place` / `run_coordinated_retreat` 接口，复用统一 free-space facade；支撑臂在公共移动及撤退中保持测得关节状态。插入使用原有 interface 与实测 lower-to-TCP 变换，持续检查持砖、实际支撑和精确连接；接触超时由 20 s 修正为 120 s，以覆盖原有 80 mm / 1 mm/s 行程。
- 连接之后分别证明真实开爪释放、MoveIt detach/world 恢复和实际向下撤退。支撑保持至撤退及连接稳定取证结束，之后才结束支撑角色并发布 placed。插入未连接时只允许一次安全终态后的受监控卸载，不释放、不压入重试。
- 接触场景修改保留底板、地面、其他结构砖和散砖；只允许指定的指尖/上砖及连接砖对接触，不能放开整臂碰撞。
- pre-phase 验证允许本次场景同步造成的已绑定 version 更新，拒绝额外版本漂移；实际到达参与者必须与请求一致。取消委托已有会话/动作边界，不再伪造“没有运动，因此安全”。

## 验证与边界

- 最终聚焦回归：500 passed，1 skipped；覆盖新技能契约、scheduler、facade、PICK、接触执行器、ACM、runtime session 与原有 targets。
- 修改后的 SUPPORT_TOP / PLACE_UP / contact / 新辅助模块、facade、unified scene 和新测试通过 Ruff。
- 任务 C 原始 `robot-operation-plan` 输入预检：`input_valid=true`，退出码 0，`motion_sent=false`；状态 `PREFLIGHT_ONLY`，不是执行成功。
- 全量回归另有 27 项未通过，集中于 HANDOVER 几何和夹爪 mimic 契约。本轮未修改这些失败断言对应的功能，不将全量套件称为通过。
- 本轮没有启动新的 Isaac/ROS 整链运动，不能宣称新的侧向 PICK、SUPPORT_TOP 或 PLACE_UP 已经真实成功。

## 下一次执行

保持原始 `piper-run --robot-operation-plan` 入口与现有任务/配置，在新的 `runs/` 目录验证：PLACE_UP 专用 PICK → 同会话双臂公共区域 → 顶部实际支撑 → 向上插入 → 精确连接 → 释放 → 携物臂向下撤退。

重点检查新抓取在地面位置的实际 IK/整手碰撞、携物运动中的刚体跟随、支撑反力是否稳定、插入后连接与释放/撤退证据。若失败，按具体规划/物理证据定位，不能以关闭碰撞或伪造 holding/supporting 来宣称完成。
