// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rocobrick::region_binding
{
inline constexpr const char* PIPELINE = "rocobrick_region";
bool isRegion(const std::string& name);

// Real constraint messages are the key, not geometry fields repurposed as metadata.
// Entries are request-scoped and scene-specific; OMPL worker threads share the registry.
struct Entry
{
  const planning_scene::PlanningScene* scene;
  std::string group;
  moveit_msgs::msg::Constraints constraints;
  std::string binding;
  std::atomic<unsigned int> selected{0};
};
class Registry
{
public:
  std::shared_ptr<Entry> find(const planning_scene::PlanningSceneConstPtr& scene,
                              const std::string& group,
                              const moveit_msgs::msg::Constraints& constraints);
  void insert(const std::vector<std::shared_ptr<Entry>>& entries);
  void erase(const std::vector<std::shared_ptr<Entry>>& entries) noexcept;
private:
  std::mutex mutex_;
  std::vector<std::shared_ptr<Entry>> entries_;
};
// Defined once in the same shared library as both planner and allocator plugins.
Registry& registry();

class Scope
{
public:
  Scope(const planning_scene::PlanningSceneConstPtr& scene,
        const moveit_msgs::msg::MotionPlanRequest& request);
  ~Scope();
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  bool allSelected() const;
  std::size_t expected() const { return entries_.size(); }
  std::size_t selected() const;
private:
  planning_scene::PlanningSceneConstPtr scene_;
  std::vector<std::shared_ptr<Entry>> entries_;
};
}  // namespace rocobrick::region_binding
