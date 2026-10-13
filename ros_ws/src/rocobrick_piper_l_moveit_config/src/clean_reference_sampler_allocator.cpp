// SPDX-License-Identifier: BSD-3-Clause
#include <rocobrick_piper_l_moveit_config/clean_reference_union_sampler.hpp>
#include <rocobrick_piper_l_moveit_config/handover_region_sampler.hpp>
#include <rocobrick_piper_l_moveit_config/grasp_region_sampler.hpp>
#include <moveit/constraint_samplers/constraint_sampler_manager.h>
#include <pluginlib/class_list_macros.hpp>
#include <set>

namespace rocobrick
{
class CleanReferenceSamplerAllocator final : public constraint_samplers::ConstraintSamplerAllocator
{
public:
  bool canService(const planning_scene::PlanningSceneConstPtr& scene, const std::string& group,
                  const moveit_msgs::msg::Constraints& constraints) const override
  {
    if (constraints.name.rfind("rocobrick-grasp-region/v1;", 0) == 0)
      return scene && (group == "dual_arm" || group == "dual_arm_tcp") &&
             scene->getRobotModel()->getJointModelGroup(group) &&
             constraints.visibility_constraints.empty();
    // Only the combined calibrated dual-TCP goal uses this compatibility fix.
    // Joint-only PICK/lift and individual-arm planning retain default samplers.
    if (!scene || (group != "dual_arm" && group != "dual_arm_tcp") ||
        !scene->getRobotModel()->getJointModelGroup(group) ||
        !constraints.joint_constraints.empty() || !constraints.visibility_constraints.empty() ||
        constraints.position_constraints.size() != 2 || constraints.orientation_constraints.size() != 2)
      return false;
    const std::set<std::string> expected = { "piper_0_rocobrick_tcp", "piper_1_rocobrick_tcp" };
    std::set<std::string> positions, orientations;
    for (const auto& constraint : constraints.position_constraints)
      positions.insert(constraint.link_name);
    for (const auto& constraint : constraints.orientation_constraints)
      orientations.insert(constraint.link_name);
    return positions == expected && orientations == expected;
  }

  constraint_samplers::ConstraintSamplerPtr alloc(const planning_scene::PlanningSceneConstPtr& scene,
                                                  const std::string& group,
                                                  const moveit_msgs::msg::Constraints& constraints) override
  {
    if (!canService(scene, group, constraints))
      return {};
    if (constraints.name.rfind("rocobrick-grasp-region/v1;", 0) == 0)
    {
      auto sampler = std::make_shared<GraspRegionSampler>(scene, group);
      return sampler->configure(constraints) ? sampler : constraint_samplers::ConstraintSamplerPtr{};
    }
    if (constraints.name.rfind("rocobrick-handover-region/v1;", 0) == 0)
    {
      auto sampler = std::make_shared<HandoverRegionSampler>(scene, group);
      return sampler->configure(constraints) ? sampler : constraint_samplers::ConstraintSamplerPtr{};
    }
    // The static default selector cannot recurse into this plugin allocator.
    // Preserve its IK configuration and subgroup dependency ordering verbatim.
    const auto selected = constraint_samplers::ConstraintSamplerManager::selectDefaultSampler(scene, group, constraints);
    const auto combined = std::dynamic_pointer_cast<constraint_samplers::UnionConstraintSampler>(selected);
    if (!combined || combined->getSamplers().size() != 2)
      return {};  // No manufactured IK capability or silent joint fallback.
    return std::make_shared<CleanReferenceUnionSampler>(scene, group, combined->getSamplers());
  }
};
}  // namespace rocobrick

PLUGINLIB_EXPORT_CLASS(rocobrick::CleanReferenceSamplerAllocator, constraint_samplers::ConstraintSamplerAllocator)
