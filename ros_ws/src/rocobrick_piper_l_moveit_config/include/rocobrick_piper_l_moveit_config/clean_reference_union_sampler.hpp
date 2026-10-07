// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <moveit/constraint_samplers/union_constraint_sampler.h>

namespace rocobrick
{
// Humble's UnionConstraintSampler randomizes the output before the first IK
// sampler reads its reference. OMPL calls sample(state, attempts), aliasing
// those states. Preserve a clean reference before delegating; keep upstream
// ordering, IK, retries, and the planner's final constraint/collision checks.
class CleanReferenceUnionSampler final : public constraint_samplers::UnionConstraintSampler
{
public:
  CleanReferenceUnionSampler(const planning_scene::PlanningSceneConstPtr& scene,
                            const std::string& group,
                            const std::vector<constraint_samplers::ConstraintSamplerPtr>& samplers)
    : UnionConstraintSampler(scene, group, samplers)
  {
    is_valid_ = jmg_ != nullptr && !samplers.empty();
  }

  using constraint_samplers::ConstraintSampler::sample;

  bool sample(moveit::core::RobotState& state, const moveit::core::RobotState& reference,
              unsigned int attempts) override
  {
    if (!is_valid_)
      return false;
    moveit::core::RobotState stable_reference(reference);
    stable_reference.updateLinkTransforms();
    return UnionConstraintSampler::sample(state, stable_reference, attempts);
  }

  const std::string& getName() const override
  {
    static const std::string name = "RoCoCleanReferenceUnionSampler";
    return name;
  }
};
}  // namespace rocobrick
