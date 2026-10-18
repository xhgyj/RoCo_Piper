// SPDX-License-Identifier: BSD-3-Clause
#include <rocobrick_piper_l_moveit_config/region_planning_context.hpp>
#include <rocobrick_piper_l_moveit_config/grasp_region_sampler.hpp>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/constraint_samplers/constraint_sampler_manager.h>
#include <pluginlib/class_loader.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

namespace rb = rocobrick::region_binding;
namespace
{
planning_scene::PlanningScenePtr scene()
{
  auto urdf = urdf::parseURDF(R"(<robot name="binding_test"><link name="root"/><link name="tip"/>
    <joint name="ja" type="prismatic"><parent link="root"/><child link="tip"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>)");
  auto srdf = std::make_shared<srdf::Model>();
  srdf->initString(*urdf, R"(<robot name="binding_test"><group name="dual_arm"><joint name="ja"/></group></robot>)");
  return std::make_shared<planning_scene::PlanningScene>(std::make_shared<moveit::core::RobotModel>(urdf, srdf));
}
moveit_msgs::msg::MotionPlanRequest request()
{
  moveit_msgs::msg::MotionPlanRequest req;
  req.pipeline_id = rb::PIPELINE;
  req.group_name = "dual_arm";
  moveit_msgs::msg::Constraints goal;
  goal.name = "rocobrick-grasp-region/v1;test-binding";
  moveit_msgs::msg::JointConstraint joint;
  joint.joint_name = "ja"; joint.position = .5; joint.tolerance_above = .1; joint.tolerance_below = .1; joint.weight = 1;
  goal.joint_constraints.push_back(joint);
  req.goal_constraints.push_back(goal);
  joint.tolerance_above = joint.tolerance_below = .05;
  req.path_constraints.joint_constraints.push_back(joint);
  return req;
}
moveit_msgs::msg::Constraints roundTrip(const planning_scene::PlanningScenePtr& sc,
                                       const moveit_msgs::msg::MotionPlanRequest& req, std::size_t index = 0)
{
  kinematic_constraints::KinematicConstraintSet set(sc->getRobotModel());
  EXPECT_TRUE(set.add(kinematic_constraints::mergeConstraints(req.goal_constraints[index], req.path_constraints),
                      sc->getTransforms()));
  return set.getAllConstraints();
}
TEST(RegionBinding, SurvivesRealMoveItMergeAndReconstructionThenExpires)
{
  auto sc = scene(); auto req = request(); auto key = roundTrip(sc, req);
  ASSERT_TRUE(key.name.empty());
  ASSERT_NE(key.joint_constraints.front().tolerance_above, req.goal_constraints.front().joint_constraints.front().tolerance_above);
  {
    rb::Scope scope(sc, req);
    auto entry = rb::registry().find(sc, req.group_name, key);
    ASSERT_TRUE(entry); EXPECT_EQ(entry->binding, req.goal_constraints.front().name);
    EXPECT_FALSE(scope.allSelected()); ++entry->selected; EXPECT_TRUE(scope.allSelected());
  }
  EXPECT_FALSE(rb::registry().find(sc, req.group_name, key));
}
TEST(RegionBinding, RejectsMissingPipelineBindingsInvalidAndAmbiguousGoals)
{
  auto sc = scene(); auto req = request();
  req.pipeline_id.clear(); EXPECT_THROW(rb::Scope(sc, req), std::invalid_argument);
  req = request(); req.goal_constraints.front().name.clear(); EXPECT_THROW(rb::Scope(sc, req), std::invalid_argument);
  req = request(); req.goal_constraints.front().joint_constraints.front().joint_name = "absent";
  EXPECT_THROW(rb::Scope(sc, req), std::invalid_argument);
  req = request(); req.goal_constraints.push_back(req.goal_constraints.front());
  req.goal_constraints.back().name += "-different"; EXPECT_THROW(rb::Scope(sc, req), std::invalid_argument);
  EXPECT_FALSE(rb::registry().find(sc, "dual_arm", roundTrip(sc, request())));
}
TEST(RegionBinding, IsolatesScenesGroupsAndConcurrentLookups)
{
  auto a = scene(); auto b = scene(); auto req = request(); auto key = roundTrip(a, req);
  rb::Scope sa(a, req); rb::Scope sb(b, req);
  EXPECT_THROW(rb::Scope(a, req), std::runtime_error);
  EXPECT_FALSE(rb::registry().find(a, "other", key));
  auto ea = rb::registry().find(a, "dual_arm", key); auto eb = rb::registry().find(b, "dual_arm", key);
  ASSERT_TRUE(ea); ASSERT_TRUE(eb); EXPECT_NE(ea, eb);
  std::thread first([&] { for (int i = 0; i < 100; ++i) ++rb::registry().find(a, "dual_arm", key)->selected; });
  std::thread second([&] { for (int i = 0; i < 100; ++i) ++rb::registry().find(b, "dual_arm", key)->selected; });
  first.join(); second.join(); EXPECT_EQ(ea->selected, 100u); EXPECT_EQ(eb->selected, 100u);
}
TEST(RegionBinding, RequiresEveryDistinctGoalAndSupportsBothSchemas)
{
  auto sc = scene(); auto req = request(); req.goal_constraints.push_back(req.goal_constraints.front());
  req.goal_constraints.back().name = "rocobrick-handover-region/v1;second";
  req.goal_constraints.back().joint_constraints.front().position = .55;
  rb::Scope scope(sc, req); EXPECT_EQ(scope.expected(), 2u);
  ++rb::registry().find(sc, "dual_arm", roundTrip(sc, req))->selected;
  EXPECT_FALSE(scope.allSelected());
  ++rb::registry().find(sc, "dual_arm", roundTrip(sc, req, 1))->selected;
  EXPECT_TRUE(scope.allSelected());
}
class StubContext final : public planning_interface::PlanningContext
{
public:
  explicit StubContext(const planning_scene::PlanningScenePtr& sc, bool throws = false)
    : PlanningContext("stub", "dual_arm"), throws_(throws) { setPlanningScene(sc); setMotionPlanRequest(request()); }
  bool solve(planning_interface::MotionPlanResponse& response) override
  {
    if (throws_) throw std::runtime_error("test planner exception");
    response.trajectory_ = std::make_shared<robot_trajectory::RobotTrajectory>(getPlanningScene()->getRobotModel(), "dual_arm");
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS; return true;
  }
  bool solve(planning_interface::MotionPlanDetailedResponse& response) override
  {
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    response.trajectory_.push_back(std::make_shared<robot_trajectory::RobotTrajectory>(getPlanningScene()->getRobotModel(), "dual_arm"));
    return true;
  }
  bool terminate() override { return true; }
  void clear() override {}
private:
  bool throws_;
};
TEST(RegionBinding, ClearsSuccessfulFallbackTrajectoryAndCleansExceptions)
{
  for (bool throws : {false, true})
  {
    auto sc = scene(); auto req = request(); auto key = roundTrip(sc, req);
    rocobrick::RegionPlanningContext context(std::make_shared<StubContext>(sc, throws), std::make_shared<rb::Scope>(sc, req));
    planning_interface::MotionPlanResponse response;
    EXPECT_FALSE(context.solve(response)); EXPECT_FALSE(response.trajectory_);
    EXPECT_EQ(response.error_code_.val, moveit_msgs::msg::MoveItErrorCodes::FAILURE);
    EXPECT_FALSE(rb::registry().find(sc, "dual_arm", key));
  }
}
TEST(RegionBinding, DetailedFallbackAlsoFailsClosedAndSelectedPlanCanPass)
{
  auto sc = scene(); auto req = request(); auto key = roundTrip(sc, req);
  {
    rocobrick::RegionPlanningContext context(std::make_shared<StubContext>(sc), std::make_shared<rb::Scope>(sc, req));
    planning_interface::MotionPlanDetailedResponse response;
    EXPECT_FALSE(context.solve(response)); EXPECT_TRUE(response.trajectory_.empty());
    EXPECT_EQ(response.error_code_.val, moveit_msgs::msg::MoveItErrorCodes::FAILURE);
  }
  rocobrick::RegionPlanningContext context(std::make_shared<StubContext>(sc), std::make_shared<rb::Scope>(sc, req));
  ++rb::registry().find(sc, "dual_arm", key)->selected;
  planning_interface::MotionPlanResponse response;
  EXPECT_TRUE(context.solve(response)); EXPECT_TRUE(response.trajectory_);
  EXPECT_FALSE(rb::registry().find(sc, "dual_arm", key));
  EXPECT_FALSE(context.solve(response)); EXPECT_FALSE(response.trajectory_);  // Never reuse a spent scope.
}
std::string readFile(const std::string& path)
{
  std::ifstream input(path); if (!input) throw std::runtime_error("missing model fixture: " + path);
  std::ostringstream stream; stream << input.rdbuf(); return stream.str();
}
TEST(RegionBindingInstalledPlugins, RealPiperKdlAndAllocatorSelectAfterNameLoss)
{
  const auto* dir = std::getenv("ROCOBRICK_TEST_MODEL_DIR");
  if (!dir) GTEST_SKIP() << "Set ROCOBRICK_TEST_MODEL_DIR to a generated dual-arm model directory";
  if (!rclcpp::ok()) { int argc = 0; rclcpp::init(argc, nullptr); }
  auto node = std::make_shared<rclcpp::Node>("region_binding_test");
  node->declare_parameter("constraint_samplers", "rocobrick/CleanReferenceSamplerAllocator");
  pluginlib::ClassLoader<planning_interface::PlannerManager> planners("moveit_core", "planning_interface::PlannerManager");
  pluginlib::ClassLoader<kinematics::KinematicsBase> kdl("moveit_core", "kinematics::KinematicsBase");
  pluginlib::ClassLoader<constraint_samplers::ConstraintSamplerAllocator> allocators(
      "moveit_core", "constraint_samplers::ConstraintSamplerAllocator");
  auto urdf = urdf::parseURDF(readFile(std::string(dir) + "/model.urdf")); ASSERT_TRUE(urdf);
  auto srdf = std::make_shared<srdf::Model>(); ASSERT_TRUE(srdf->initString(*urdf, readFile(std::string(dir) + "/model.srdf")));
  auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
  std::map<std::string, moveit::core::SolverAllocatorFn> solvers;
  for (const std::string robot : {"piper_0", "piper_1"})
    solvers[robot + "_arm_tcp"] = [&, robot](const moveit::core::JointModelGroup* group) {
      auto solver = kdl.createSharedInstance("kdl_kinematics_plugin/KDLKinematicsPlugin");
      if (!solver->initialize(node, *model, group->getName(), robot + "_base_link", {robot + "_rocobrick_tcp"}, .005))
        throw std::runtime_error("KDL initialization failed");
      return solver;
    };
  model->setKinematicsAllocators(solvers);
  auto sc = std::make_shared<planning_scene::PlanningScene>(model);
  moveit_msgs::msg::MotionPlanRequest req; req.pipeline_id = rb::PIPELINE; req.group_name = "dual_arm";
  moveit_msgs::msg::Constraints goal;
  goal.name = "rocobrick-grasp-region/v1;1;piper_0;" + std::string(64, 'a') +
      ";2;0;0;0;0;0;0;1;0;0;0;0;0;0;1;0;0;0;.01;.01;.01;0;.005;.001;.1305896;.06;-.0127;0;-.0048;.0127;0;.0052";
  moveit_msgs::msg::PositionConstraint pc; pc.header.frame_id = "piper_0_base_link"; pc.link_name = "piper_0_rocobrick_tcp"; pc.weight = 1;
  shape_msgs::msg::SolidPrimitive sphere; sphere.type = sphere.SPHERE; sphere.dimensions = {.02};
  geometry_msgs::msg::Pose pose; pose.orientation.w = 1; pose.position.z = .1;
  pc.constraint_region.primitives.push_back(sphere); pc.constraint_region.primitive_poses.push_back(pose); goal.position_constraints.push_back(pc);
  moveit_msgs::msg::OrientationConstraint oc; oc.header = pc.header; oc.link_name = pc.link_name; oc.orientation.w = 1;
  oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = M_PI; oc.weight = 1;
  goal.orientation_constraints.push_back(oc);
  for (const auto& name : model->getJointModelGroup("piper_1_arm_tcp")->getVariableNames())
  {
    moveit_msgs::msg::JointConstraint jc; jc.joint_name = name; jc.position = sc->getCurrentState().getVariablePosition(name);
    jc.tolerance_above = jc.tolerance_below = .002; jc.weight = 1;
    goal.joint_constraints.push_back(jc); req.path_constraints.joint_constraints.push_back(jc);
  }
  req.goal_constraints.push_back(goal); auto key = roundTrip(sc, req); ASSERT_TRUE(key.name.empty());
  constraint_samplers::ConstraintSamplerManager manager;
  manager.registerSamplerAllocator(allocators.createSharedInstance("rocobrick/CleanReferenceSamplerAllocator"));
  EXPECT_FALSE(manager.selectSampler(sc, "dual_arm", key));  // Missing scope must NOT use default IK.
  {
    rb::Scope scope(sc, req);
    auto peer = manager.selectSampler(sc, "dual_arm", req.path_constraints); ASSERT_TRUE(peer);
    EXPECT_FALSE(scope.allSelected());
    auto selected = manager.selectSampler(sc, "dual_arm", key); ASSERT_TRUE(selected); EXPECT_TRUE(selected->isValid());
    EXPECT_TRUE(std::dynamic_pointer_cast<rocobrick::GraspRegionSampler>(selected));
    EXPECT_TRUE(scope.allSelected());
    EXPECT_EQ(rb::registry().find(sc, "dual_arm", key)->binding, goal.name);
  }
  EXPECT_FALSE(manager.selectSampler(sc, "dual_arm", key));
  auto planner = planners.createSharedInstance("rocobrick/RegionPlannerManager");
  ASSERT_TRUE(planner->initialize(model, node, rb::PIPELINE));
  req.allowed_planning_time = .1;
  moveit_msgs::msg::MoveItErrorCodes error;
  auto context = planner->getPlanningContext(sc, req, error);
  ASSERT_TRUE(context) << "error=" << error.val;
  auto entry = rb::registry().find(sc, "dual_arm", key);
  ASSERT_TRUE(entry); EXPECT_GT(entry->selected.load(), 0u);
  context.reset();
  EXPECT_FALSE(rb::registry().find(sc, "dual_arm", key));
  auto missing = req; missing.goal_constraints.front().name.clear();
  EXPECT_FALSE(planner->getPlanningContext(sc, missing, error));
  EXPECT_EQ(error.val, moveit_msgs::msg::MoveItErrorCodes::FAILURE);
  missing = req; missing.pipeline_id.clear();
  EXPECT_FALSE(planner->getPlanningContext(sc, missing, error));
}
}  // namespace
