// SPDX-License-Identifier: BSD-3-Clause
#include <rocobrick_piper_l_moveit_config/clean_reference_union_sampler.hpp>
#include <rocobrick_piper_l_moveit_config/handover_region_sampler.hpp>
#include <rocobrick_piper_l_moveit_config/grasp_region_sampler.hpp>
#include <gtest/gtest.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

namespace
{
TEST(GraspRegionDiagnostics, RejectionsAndAcceptanceReconcile)
{
  rocobrick::grasp_diag::Counters counters;
  counters.calls = 1;
  counters.attempts_budget = 4;
  counters.attempts_run = 3;
  counters.reject(rocobrick::grasp_diag::kGeometry);
  counters.reject(rocobrick::grasp_diag::kIk);
  counters.accepted = 1;
  counters.accept_attempt_sum = 3;
  const auto summary = counters.summary();
  EXPECT_NE(summary.find("attempts_run=3 accepted=1 first_reject_total=2"), std::string::npos);
  EXPECT_NE(summary.find("accepted_attempt_avg=3"), std::string::npos);
  EXPECT_EQ(counters.first[rocobrick::grasp_diag::kGeometry], 1u);
  EXPECT_EQ(counters.first[rocobrick::grasp_diag::kIk], 1u);
}

TEST(GraspRegionDiagnostics, AttemptPreservesFirstRejection)
{
  rocobrick::grasp_diag::Attempt attempt;
  attempt.note(rocobrick::grasp_diag::kGeometry);
  attempt.note(rocobrick::grasp_diag::kIk);
  EXPECT_EQ(attempt.stage, rocobrick::grasp_diag::kGeometry);
}

TEST(GraspRegion, DecodesContinuousRegionAndRejectsMalformedBindings)
{
  std::vector<rocobrick::GraspRegionSampler::Region> regions;
  const std::string binding = "rocobrick-grasp-region/v1;1;piper_0;" + std::string(64, 'a') +
      ";2;0;0;0;0;0;0;1;0;0;0;0;0;0;1;0;0;0;.01;.01;.01;0;.005;.001;.1305896;.06;-.0127;0;-.0048;.0127;0;.0052";
  ASSERT_TRUE(rocobrick::GraspRegionSampler::decode(binding, regions));
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions[0].robot, "piper_0");
  EXPECT_EQ(regions[0].pads.size(), 2u);
  EXPECT_FALSE(rocobrick::GraspRegionSampler::decode(binding + ";extra", regions));
  EXPECT_FALSE(rocobrick::GraspRegionSampler::decode("rocobrick-grasp-region/v1;3", regions));
  EXPECT_FALSE(rocobrick::GraspRegionSampler::decode("rocobrick-grasp-region/v1;1;piper_2", regions));
}

TEST(GraspRegion, ContactGeometryChecksRealPadFloorAndStudCorridor)
{
  rocobrick::GraspRegionSampler::Region region;
  region.center = Eigen::Vector3d(0, 0, .01);
  region.half = Eigen::Vector3d(.01, .01, .01);
  region.floor_margin = .005;
  region.contact_margin = .001;
  region.flange = .1305896;
  region.opening = .06;
  region.pads = {Eigen::Vector3d(-.0127, 0, -.0048), Eigen::Vector3d(.0127, 0, .0052)};
  Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
  local.linear() = Eigen::AngleAxisd(M_PI / 2., Eigen::Vector3d::UnitY()).toRotationMatrix();
  local.translation() = Eigen::Vector3d(0, 0, .018);
  EXPECT_TRUE(rocobrick::GraspRegionSampler::contactGeometry(region, Eigen::Isometry3d::Identity(), local));
  local.translation().z() = .01;
  EXPECT_FALSE(rocobrick::GraspRegionSampler::contactGeometry(region, Eigen::Isometry3d::Identity(), local));
  local.translation().z() = .018;
  local.linear() = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()).toRotationMatrix();
  EXPECT_FALSE(rocobrick::GraspRegionSampler::contactGeometry(region, Eigen::Isometry3d::Identity(), local));
}

planning_scene::PlanningScenePtr makeScene()
{
  const auto urdf = urdf::parseURDF(R"(
    <robot name="sampler_test">
      <link name="root"/><link name="a"/><link name="b"/><link name="other"/>
      <joint name="ja" type="prismatic"><parent link="root"/><child link="a"/>
        <axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
      <joint name="jb" type="prismatic"><parent link="root"/><child link="b"/>
        <axis xyz="0 1 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
      <joint name="outside" type="prismatic"><parent link="root"/><child link="other"/>
        <axis xyz="0 0 1"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
    </robot>)");
  auto srdf = std::make_shared<srdf::Model>();
  srdf->initString(*urdf, R"(<robot name="sampler_test">
    <group name="a_group"><joint name="ja"/></group>
    <group name="b_group"><joint name="jb"/></group>
    <group name="dual_arm"><group name="a_group"/><group name="b_group"/></group>
    </robot>)");
  auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
  return std::make_shared<planning_scene::PlanningScene>(model);
}

class RequiresCleanReference final : public constraint_samplers::ConstraintSampler
{
public:
  RequiresCleanReference(const planning_scene::PlanningSceneConstPtr& scene, const std::string& group,
                         double target, bool succeeds = true)
    : ConstraintSampler(scene, group), target_(target), succeeds_(succeeds)
  {
    is_valid_ = true;
  }
  bool configure(const moveit_msgs::msg::Constraints&) override { return true; }
  bool sample(moveit::core::RobotState& state, const moveit::core::RobotState& reference,
              unsigned int attempts) override
  {
    last_attempts = attempts;
    if (reference.dirtyLinkTransforms() || !succeeds_)
      return false;
    reference_a = reference.getVariablePosition("ja");
    state.setJointGroupPositions(jmg_, std::vector<double>{target_});
    return true;
  }
  bool project(moveit::core::RobotState& state, unsigned int attempts) override
  {
    return sample(state, state, attempts);
  }
  const std::string& getName() const override
  {
    static const std::string name = "RequiresCleanReference";
    return name;
  }
  double reference_a = -1.0;
  unsigned int last_attempts = 0;
private:
  double target_;
  bool succeeds_;
};

struct SamplerFixture : testing::Test
{
  planning_scene::PlanningScenePtr scene = makeScene();
  moveit::core::RobotState state = scene->getCurrentState();
  std::shared_ptr<RequiresCleanReference> a = std::make_shared<RequiresCleanReference>(scene, "a_group", 0.4);
  std::shared_ptr<RequiresCleanReference> b = std::make_shared<RequiresCleanReference>(scene, "b_group", 0.6);
  std::vector<constraint_samplers::ConstraintSamplerPtr> children{b, a};
  SamplerFixture()
  {
    state.setVariablePosition("ja", 0.1);
    state.setVariablePosition("jb", 0.2);
    state.setVariablePosition("outside", 0.3);
    state.update();
  }
};

TEST_F(SamplerFixture, OriginalAliasCallReproducesDirtyReferenceFailure)
{
  constraint_samplers::ConstraintSamplerPtr original =
      std::make_shared<constraint_samplers::UnionConstraintSampler>(scene, "dual_arm", children);
  ASSERT_FALSE(state.dirtyLinkTransforms());
  EXPECT_FALSE(original->sample(state, 17));
}

TEST_F(SamplerFixture, AliasCallKeepsCleanReferenceAndSubgroupOrder)
{
  constraint_samplers::ConstraintSamplerPtr fixed =
      std::make_shared<rocobrick::CleanReferenceUnionSampler>(scene, "dual_arm", children);
  ASSERT_TRUE(fixed->isValid());
  ASSERT_TRUE(fixed->sample(state, 17));
  EXPECT_DOUBLE_EQ(a->reference_a, 0.1);  // Before randomization, not aliased output.
  EXPECT_DOUBLE_EQ(b->reference_a, 0.4);  // Second subgroup sees the first's solution.
  EXPECT_EQ(a->last_attempts, 17u);
  EXPECT_EQ(b->last_attempts, 17u);
  EXPECT_DOUBLE_EQ(state.getVariablePosition("ja"), 0.4);
  EXPECT_DOUBLE_EQ(state.getVariablePosition("jb"), 0.6);
  EXPECT_DOUBLE_EQ(state.getVariablePosition("outside"), 0.3);
}

TEST_F(SamplerFixture, DirtyIncomingReferenceIsUpdatedWithoutMutatingCaller)
{
  auto reference = state;
  reference.setVariablePosition("ja", 0.25);
  ASSERT_TRUE(reference.dirtyLinkTransforms());
  rocobrick::CleanReferenceUnionSampler fixed(scene, "dual_arm", children);
  ASSERT_TRUE(fixed.sample(state, reference, 9));
  EXPECT_DOUBLE_EQ(a->reference_a, 0.25);
  EXPECT_TRUE(reference.dirtyLinkTransforms());
  EXPECT_DOUBLE_EQ(reference.getVariablePosition("ja"), 0.25);
}

TEST_F(SamplerFixture, RepeatedOmplStyleCallsDoNotLeaveReferenceDirty)
{
  constraint_samplers::ConstraintSamplerPtr fixed =
      std::make_shared<rocobrick::CleanReferenceUnionSampler>(scene, "dual_arm", children);
  for (int i = 0; i < 20; ++i)
    ASSERT_TRUE(fixed->sample(state, 5));
}

TEST_F(SamplerFixture, ChildFailureIsNotTurnedIntoSuccess)
{
  auto fails = std::make_shared<RequiresCleanReference>(scene, "b_group", 0.6, false);
  rocobrick::CleanReferenceUnionSampler fixed(scene, "dual_arm", {a, fails});
  EXPECT_FALSE(fixed.sample(state, 11));
}

TEST_F(SamplerFixture, ProjectPreservesUpstreamTransformUpdates)
{
  rocobrick::CleanReferenceUnionSampler fixed(scene, "dual_arm", children);
  state.setVariablePosition("ja", 0.25);
  EXPECT_TRUE(fixed.project(state, 3));
  EXPECT_DOUBLE_EQ(state.getVariablePosition("ja"), 0.4);
  EXPECT_DOUBLE_EQ(state.getVariablePosition("jb"), 0.6);
}

TEST_F(SamplerFixture, EmptyUnionDoesNotAdvertiseCapability)
{
  rocobrick::CleanReferenceUnionSampler fixed(scene, "dual_arm", {});
  EXPECT_FALSE(fixed.isValid());
  EXPECT_FALSE(fixed.sample(state, 1));
}

TEST_F(SamplerFixture, RegionRejectsMalformedMetadataWithoutInventingIkCapability)
{
  rocobrick::HandoverRegionSampler sampler(scene, "dual_arm");
  moveit_msgs::msg::Constraints constraints;
  for (const auto& name : {"", "rocobrick-handover-region/v1;piper_0",
                          "rocobrick-handover-region/v1;piper_2;bad"})
  {
    constraints.name = name;
    EXPECT_FALSE(sampler.configure(constraints));
    EXPECT_FALSE(sampler.isValid());
    EXPECT_FALSE(sampler.sample(state, 1));
  }
}
}  // namespace
