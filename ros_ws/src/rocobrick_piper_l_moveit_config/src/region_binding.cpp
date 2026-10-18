// SPDX-License-Identifier: BSD-3-Clause
#include <rocobrick_piper_l_moveit_config/region_binding.hpp>
#include <moveit/kinematic_constraints/kinematic_constraint.h>
#include <moveit/kinematic_constraints/utils.h>
#include <algorithm>
#include <stdexcept>

namespace rocobrick::region_binding
{
bool isRegion(const std::string& name)
{
  return name.rfind("rocobrick-grasp-region/v1;", 0) == 0 ||
         name.rfind("rocobrick-handover-region/v1;", 0) == 0;
}
Registry& registry()
{
  static Registry instance;
  return instance;
}
std::shared_ptr<Entry> Registry::find(const planning_scene::PlanningSceneConstPtr& scene,
                                    const std::string& group,
                                    const moveit_msgs::msg::Constraints& constraints)
{
  auto key = constraints;
  key.name.clear();
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& entry : entries_)
    if (entry->scene == scene.get() && entry->group == group && entry->constraints == key)
      return entry;
  return {};
}
void Registry::insert(const std::vector<std::shared_ptr<Entry>>& entries)
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& candidate : entries)
    for (const auto& existing : entries_)
      if (candidate->scene == existing->scene)
        throw std::runtime_error("overlapping region binding scope for the same planning scene");
  // Strong exception guarantee: a failed allocation must not leave a partial registration.
  auto next = entries_;
  next.insert(next.end(), entries.begin(), entries.end());
  entries_.swap(next);
}
void Registry::erase(const std::vector<std::shared_ptr<Entry>>& entries) noexcept
{
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const auto& entry) {
    return std::find(entries.begin(), entries.end(), entry) != entries.end();
  }), entries_.end());
}
Scope::Scope(const planning_scene::PlanningSceneConstPtr& scene,
             const moveit_msgs::msg::MotionPlanRequest& request) : scene_(scene)
{
  if (!scene || request.pipeline_id != PIPELINE || request.goal_constraints.empty() ||
      (request.group_name != "dual_arm" && request.group_name != "dual_arm_tcp"))
    throw std::invalid_argument("region requests require the dedicated pipeline and combined group");
  for (const auto& goal : request.goal_constraints)
  {
    if (!isRegion(goal.name))
      throw std::invalid_argument("region pipeline rejects missing, unknown or mixed goal bindings");
    const auto merged = kinematic_constraints::mergeConstraints(goal, request.path_constraints);
    kinematic_constraints::KinematicConstraintSet set(scene->getRobotModel());
    if (!set.add(merged, scene->getTransforms()) || set.empty())
      throw std::invalid_argument("invalid or empty region constraints");
    auto entry = std::make_shared<Entry>();
    entry->scene = scene.get();
    entry->group = request.group_name;
    entry->constraints = set.getAllConstraints();
    entry->constraints.name.clear();
    entry->binding = goal.name;
    bool duplicate = false;
    for (const auto& other : entries_)
      if (other->constraints == entry->constraints)
      {
        if (other->binding != entry->binding)
          throw std::invalid_argument("ambiguous region bindings for identical constraints");
        duplicate = true;
      }
    if (!duplicate)
      entries_.push_back(std::move(entry));
  }
  registry().insert(entries_);
}
Scope::~Scope() { registry().erase(entries_); }
std::size_t Scope::selected() const
{
  return std::count_if(entries_.begin(), entries_.end(), [](const auto& entry) {
    return entry->selected.load() > 0;
  });
}
bool Scope::allSelected() const { return !entries_.empty() && selected() == expected(); }
}  // namespace rocobrick::region_binding
