// SPDX-License-Identifier: BSD-3-Clause
#include <rocobrick_piper_l_moveit_config/region_planning_context.hpp>
#include <pluginlib/class_loader.hpp>
#include <algorithm>
#include <pluginlib/class_list_macros.hpp>

namespace rocobrick
{
class RegionPlannerManager final : public planning_interface::PlannerManager
{
public:
  bool initialize(const moveit::core::RobotModelConstPtr& model, const rclcpp::Node::SharedPtr& node,
                  const std::string& ns) override
  {
    if (ns != region_binding::PIPELINE)
      return false;
    try
    {
      loader_ = std::make_unique<pluginlib::ClassLoader<planning_interface::PlannerManager>>(
          "moveit_core", "planning_interface::PlannerManager");
      planner_ = loader_->createSharedInstance("ompl_interface/OMPLPlanner");
      if (!planner_->initialize(model, node, ns))
      {
        planner_.reset();
        return false;
      }
      config_settings_ = planner_->getPlannerConfigurations();
      return true;
    }
    catch (const std::exception& error)
    {
      planner_.reset();
      std::fprintf(stderr, "[region-binding] initialization_failed=%s\n", error.what());
      return false;
    }
  }
  std::string getDescription() const override { return "RoCo region-bound OMPL (fail closed)"; }
  void getPlanningAlgorithms(std::vector<std::string>& algorithms) const override
  {
    if (planner_)
      planner_->getPlanningAlgorithms(algorithms);
  }
  bool canServiceRequest(const planning_interface::MotionPlanRequest& request) const override
  {
    return planner_ && request.pipeline_id == region_binding::PIPELINE &&
           !request.goal_constraints.empty() &&
           std::all_of(request.goal_constraints.begin(), request.goal_constraints.end(), [](const auto& goal) {
             return region_binding::isRegion(goal.name);
           }) && planner_->canServiceRequest(request);
  }
  void setPlannerConfigurations(const planning_interface::PlannerConfigurationMap& configurations) override
  {
    if (planner_)
    {
      planner_->setPlannerConfigurations(configurations);
      config_settings_ = planner_->getPlannerConfigurations();
    }
  }
  planning_interface::PlanningContextPtr getPlanningContext(
      const planning_scene::PlanningSceneConstPtr& scene, const planning_interface::MotionPlanRequest& request,
      moveit_msgs::msg::MoveItErrorCodes& error) const override
  {
    error.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    try
    {
      if (!canServiceRequest(request))
        return {};
      auto scope = std::make_shared<region_binding::Scope>(scene, request);
      auto context = planner_->getPlanningContext(scene, request, error);
      if (!context)
      {
        if (error.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS || error.val == 0)
          error.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
        return {};
      }
      return std::make_shared<RegionPlanningContext>(std::move(context), std::move(scope));
    }
    catch (const std::exception& exception)
    {
      error.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
      std::fprintf(stderr, "[region-binding] context_rejected=%s\n", exception.what());
      return {};
    }
    catch (...)
    {
      error.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
      return {};
    }
  }
private:
  // The loader must outlive every plugin instance it creates.
  std::unique_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> loader_;
  planning_interface::PlannerManagerPtr planner_;
};
}  // namespace rocobrick
PLUGINLIB_EXPORT_CLASS(rocobrick::RegionPlannerManager, planning_interface::PlannerManager)
