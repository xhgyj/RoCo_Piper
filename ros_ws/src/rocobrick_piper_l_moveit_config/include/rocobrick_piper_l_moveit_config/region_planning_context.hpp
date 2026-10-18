// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <rocobrick_piper_l_moveit_config/region_binding.hpp>
#include <moveit/planning_interface/planning_interface.h>
#include <cstdio>

namespace rocobrick
{
// This gate is part of the planner, not an optional request adapter: MoveIt
// skips adapters that fail to load/throw, which cannot provide fail-closed routing.
class RegionPlanningContext final : public planning_interface::PlanningContext
{
public:
  RegionPlanningContext(planning_interface::PlanningContextPtr context,
                        std::shared_ptr<region_binding::Scope> scope)
    : PlanningContext(context->getName(), context->getGroupName()), context_(std::move(context)),
      scope_(std::move(scope))
  {
    setPlanningScene(context_->getPlanningScene());
    setMotionPlanRequest(context_->getMotionPlanRequest());
  }
  ~RegionPlanningContext() override
  {
    try { context_->clear(); } catch (...) {}  // Release cached native samplers before plugin loaders.
  }
  bool solve(planning_interface::MotionPlanResponse& response) override
  {
    if (!scope_)
      return refuse(response);
    try
    {
      const bool ok = context_->solve(response);
      if (!checkSelection())
        return refuse(response);
      if (!ok)
      {
        const auto code = response.error_code_.val;
        refuse(response);
        if (code != 0 && code != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
          response.error_code_.val = code;
      }
      return ok;
    }
    catch (...)
    {
      scope_.reset();
      return refuse(response);
    }
  }
  bool solve(planning_interface::MotionPlanDetailedResponse& response) override
  {
    if (!scope_)
      return refuse(response);
    try
    {
      const bool ok = context_->solve(response);
      if (!checkSelection())
        return refuse(response);
      if (!ok)
      {
        const auto code = response.error_code_.val;
        refuse(response);
        if (code != 0 && code != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
          response.error_code_.val = code;
      }
      return ok;
    }
    catch (...)
    {
      scope_.reset();
      return refuse(response);
    }
  }
  bool terminate() override { return context_->terminate(); }
  void clear() override
  {
    context_->clear();
    scope_.reset();
  }
private:
  bool checkSelection()
  {
    const bool selected = scope_->allSelected();
    std::fprintf(stderr, "[region-binding] pipeline=%s expected=%zu selected=%zu verified=%d\n",
                 region_binding::PIPELINE, scope_->expected(), scope_->selected(), selected);
    scope_.reset();
    return selected;
  }
  static bool refuse(planning_interface::MotionPlanResponse& response)
  {
    response.trajectory_.reset();
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    return false;
  }
  static bool refuse(planning_interface::MotionPlanDetailedResponse& response)
  {
    response.trajectory_.clear();
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    return false;
  }
  planning_interface::PlanningContextPtr context_;
  std::shared_ptr<region_binding::Scope> scope_;
};
}  // namespace rocobrick
