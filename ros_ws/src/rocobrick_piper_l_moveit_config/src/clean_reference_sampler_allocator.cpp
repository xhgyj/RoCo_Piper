// SPDX-License-Identifier: BSD-3-Clause
#include <rocobrick_piper_l_moveit_config/clean_reference_union_sampler.hpp>
#include <rocobrick_piper_l_moveit_config/handover_region_sampler.hpp>
#include <rocobrick_piper_l_moveit_config/grasp_region_sampler.hpp>
#include <moveit/constraint_samplers/constraint_sampler_manager.h>
#include <pluginlib/class_list_macros.hpp>
#include <set>
#include <rocobrick_piper_l_moveit_config/region_binding.hpp>

namespace rocobrick
{
// These broad free-orientation envelopes are reserved for region requests.
// Without their owning scope, claiming the request and refusing allocation is
// deliberate: ConstraintSamplerManager must not fall back to generic IK.
bool isRegionEnvelope(const moveit_msgs::msg::Constraints& constraints)
{
  for (const auto& position : constraints.position_constraints)
  {
    if (position.link_name != "piper_0_rocobrick_tcp" && position.link_name != "piper_1_rocobrick_tcp")
      continue;
    if (position.constraint_region.primitives.size() != 1 ||
        position.constraint_region.primitives.front().type != shape_msgs::msg::SolidPrimitive::SPHERE)
      continue;
    for (const auto& orientation : constraints.orientation_constraints)
      if (orientation.link_name == position.link_name &&
          orientation.absolute_x_axis_tolerance >= M_PI &&
          orientation.absolute_y_axis_tolerance >= M_PI &&
          orientation.absolute_z_axis_tolerance >= M_PI)
        return true;
  }
  return false;
}
class CleanReferenceSamplerAllocator final : public constraint_samplers::ConstraintSamplerAllocator
{
public:
  bool canService(const planning_scene::PlanningSceneConstPtr& scene, const std::string& group,
                  const moveit_msgs::msg::Constraints& constraints) const override
  {
    if (region_binding::registry().find(scene, group, constraints) || region_binding::isRegion(constraints.name) ||
        isRegionEnvelope(constraints))
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
    const auto entry = region_binding::registry().find(scene, group, constraints);
    if (entry)
    {
      auto bound = constraints;
      bound.name = entry->binding;
      constraint_samplers::ConstraintSamplerPtr sampler;
      if (bound.name.rfind("rocobrick-grasp-region/v1;", 0) == 0)
        sampler = std::make_shared<GraspRegionSampler>(scene, group);
      else if (bound.name.rfind("rocobrick-handover-region/v1;", 0) == 0)
        sampler = std::make_shared<HandoverRegionSampler>(scene, group);
      if (!sampler || !sampler->configure(bound))
        return {};  // Manager returns this failure, never selectDefaultSampler.
      ++entry->selected;
      std::fprintf(stderr, "[region-binding] allocator_selected group=%s schema=%s\n", group.c_str(),
                   bound.name.substr(0, bound.name.find(';')).c_str());
      return sampler;
    }
    if (region_binding::isRegion(constraints.name) || isRegionEnvelope(constraints))
    {
      std::fprintf(stderr, "[region-binding] rejected=missing_request_scope group=%s\n", group.c_str());
      return {};
    }  // Region metadata without its owning request scope is not capability.
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
