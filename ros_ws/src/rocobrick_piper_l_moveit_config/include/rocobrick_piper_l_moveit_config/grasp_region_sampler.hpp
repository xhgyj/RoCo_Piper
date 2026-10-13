// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <moveit/constraint_samplers/constraint_sampler_manager.h>
#include <moveit/kinematic_constraints/kinematic_constraint.h>
#include <Eigen/Geometry>
#include <array>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <map>
#include <vector>
#include <random>
#include <set>
#include <sstream>

namespace rocobrick
{
// A per-arm object-relative goal, composed into ONE dual-arm OMPL request.
// No host pose enumeration, separate arm paths or collision bypass is used.
class GraspRegionSampler final : public constraint_samplers::ConstraintSampler
{
public:
  struct Region
  {
    std::string robot;
    Eigen::Isometry3d object_base, base_world;
    Eigen::Vector3d center, half;
    std::vector<Eigen::Vector3d> pads;
    double ground = 0, floor_margin = 0, contact_margin = 0, flange = 0, opening = 0;
  };

  GraspRegionSampler(const planning_scene::PlanningSceneConstPtr& scene, const std::string& group)
    : ConstraintSampler(scene, group), rng_(std::random_device{}())
  {
  }

  static bool decode(const std::string& binding, std::vector<Region>& regions)
  {
    regions.clear();
    std::istringstream stream(binding);
    std::string token;
    auto read = [&]() { return static_cast<bool>(std::getline(stream, token, ';')); };
    auto number = [&](double& value) {
      if (!read()) return false;
      try
      {
        std::size_t used = 0;
        value = std::stod(token, &used);
        return used == token.size() && std::isfinite(value);
      }
      catch (const std::exception&) { return false; }
    };
    if (!read() || token != "rocobrick-grasp-region/v1" || !read() || (token != "1" && token != "2"))
      return false;
    const unsigned count = token == "1" ? 1 : 2;
    std::set<std::string> owners;
    for (unsigned i = 0; i < count; ++i)
    {
      Region region;
      if (!read() || (token != "piper_0" && token != "piper_1") || !owners.insert(token).second)
        return false;
      region.robot = token;
      if (!read() || token.size() != 64 || token.find_first_not_of("0123456789abcdef") != std::string::npos)
        return false;
      double vertex_count;
      if (!number(vertex_count) || vertex_count < 1 || vertex_count > 128 || std::floor(vertex_count) != vertex_count)
        return false;
      std::array<double, 25> values{};
      for (double& value : values)
        if (!number(value)) return false;
      auto pose = [&](unsigned offset, Eigen::Isometry3d& result) {
        Eigen::Quaterniond q(values[offset + 6], values[offset + 3], values[offset + 4], values[offset + 5]);
        if (std::abs(q.norm() - 1.) > 1e-6) return false;
        result = Eigen::Isometry3d::Identity();
        result.linear() = q.toRotationMatrix();
        result.translation() = Eigen::Vector3d(values[offset], values[offset + 1], values[offset + 2]);
        return true;
      };
      if (!pose(0, region.object_base) || !pose(7, region.base_world)) return false;
      region.center = Eigen::Vector3d(values[14], values[15], values[16]);
      region.half = Eigen::Vector3d(values[17], values[18], values[19]);
      region.ground = values[20];
      region.floor_margin = values[21];
      region.contact_margin = values[22];
      region.flange = values[23];
      region.opening = values[24];
      if ((region.half.array() <= 0).any() || region.floor_margin < 0 || region.contact_margin <= 0 ||
          region.contact_margin >= region.half.z() || region.flange <= 0 || region.opening <= 0 || region.opening > .1)
        return false;
      for (unsigned v = 0; v < static_cast<unsigned>(vertex_count); ++v)
      {
        Eigen::Vector3d vertex;
        for (unsigned k = 0; k < 3; ++k)
          if (!number(vertex[k])) return false;
        region.pads.push_back(vertex);
      }
      regions.push_back(std::move(region));
    }
    return !read();
  }

  bool configure(const moveit_msgs::msg::Constraints& constraints) override
  {
    is_valid_ = false;
    peers_.clear();
    groups_.clear();
    if (!jmg_ || !constraints.visibility_constraints.empty() || !decode(constraints.name, regions_))
      return false;
    const auto model = scene_->getRobotModel();
    for (const std::string robot : {"piper_0", "piper_1"})
    {
      const auto* group = model->getJointModelGroup(robot + "_arm_tcp");
      if (!group || !model->getLinkModel(robot + "_base_link") || !model->getLinkModel(robot + "_rocobrick_tcp"))
        return false;
      groups_[robot] = group;
      const auto region = std::find_if(regions_.begin(), regions_.end(), [&](const Region& r) { return r.robot == robot; });
      moveit_msgs::msg::Constraints scoped;
      for (const auto& pc : constraints.position_constraints)
        if (pc.link_name == robot + "_rocobrick_tcp") scoped.position_constraints.push_back(pc);
      for (const auto& oc : constraints.orientation_constraints)
        if (oc.link_name == robot + "_rocobrick_tcp") scoped.orientation_constraints.push_back(oc);
      const auto& variables = group->getVariableNames();
      for (const auto& jc : constraints.joint_constraints)
        if (std::find(variables.begin(), variables.end(), jc.joint_name) != variables.end()) scoped.joint_constraints.push_back(jc);
      if (region != regions_.end())
      {
        if (!group->getSolverInstance() || scoped.position_constraints.size() != 1 ||
            scoped.orientation_constraints.size() != 1 || !scoped.joint_constraints.empty())
          return false;
      }
      else
      {
        auto sampler = constraint_samplers::ConstraintSamplerManager::selectDefaultSampler(scene_, robot + "_arm_tcp", scoped);
        if (!sampler || !sampler->isValid()) return false;
        peers_.push_back(sampler);
      }
    }
    constraints_ = std::make_shared<kinematic_constraints::KinematicConstraintSet>(model);
    constraints_->add(constraints, scene_->getTransforms());
    is_valid_ = true;
    return true;
  }

  // Contact geometry only. Ground/body/model/attached collisions are still
  // independently checked by MoveIt on the sampled state and the full path.
  static bool contactGeometry(const Region& region, const Eigen::Isometry3d& object_world,
                              const Eigen::Isometry3d& local)
  {
    const Eigen::Vector3d closing = local.linear().col(1);
    const unsigned axis = std::abs(closing.x()) > std::abs(closing.y()) ? 0 : 1;
    const unsigned tangent = 1 - axis;
    const Eigen::Vector3d delta = local.translation() - region.center;
    if (std::abs(closing[axis]) < std::cos(.015) || std::abs(closing.z()) > std::sin(.015) ||
        std::abs(delta[axis]) > .001 || std::abs(delta[tangent]) > region.half[tangent] * .5 + .001 ||
        std::abs(delta.z()) > region.half.z() - region.contact_margin + .001 ||
        std::abs(local.linear()(2, 2)) >= std::cos(.015))
      return false;
    const Eigen::Vector3d flange = delta - region.flange * local.linear().col(2);
    if (std::abs(flange.x()) < region.half.x() && std::abs(flange.y()) < region.half.y() && flange.z() > region.half.z())
      return false;
    const double span = 2. * closing.cwiseAbs().dot(region.half);
    if (region.opening < span + region.contact_margin) return false;
    for (const auto& pad : region.pads)
      if ((object_world * local * pad).z() < region.ground + region.floor_margin - .001) return false;
    return true;
  }

  using constraint_samplers::ConstraintSampler::sample;
  bool sample(moveit::core::RobotState& state, const moveit::core::RobotState& reference, unsigned int attempts) override
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_valid_) return false;
    moveit::core::RobotState stable(reference);
    stable.update();
    for (unsigned attempt = 0; attempt < attempts; ++attempt)
    {
      state = stable;
      bool ok = true;
      for (const auto& peer : peers_)
      {
        moveit::core::RobotState clean(state);
        clean.update();
        if (!peer->sample(state, clean, 1)) { ok = false; break; }
        state.update();
      }
      if (!ok) continue;
      for (const auto& region : regions_)
      {
        const auto base = stable.getGlobalLinkTransform(region.robot + "_base_link");
        if ((base.translation() - region.base_world.translation()).norm() > .001 ||
            Eigen::AngleAxisd(base.linear().transpose() * region.base_world.linear()).angle() > .015)
          { ok = false; break; }
        const auto object = base * region.object_base;
        const unsigned face = std::uniform_int_distribution<unsigned>(0, 3)(rng_);
        const double pitch = std::uniform_real_distribution<double>(0., M_PI)(rng_);
        const double flip = std::uniform_int_distribution<unsigned>(0, 1)(rng_) * M_PI;
        Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
        local.linear() = (Eigen::AngleAxisd(face * M_PI / 2., Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(flip, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        const unsigned tangent = face % 2 == 0 ? 0 : 1;
        local.translation() = region.center;
        local.translation()[tangent] += std::uniform_real_distribution<double>(-region.half[tangent] * .5, region.half[tangent] * .5)(rng_);
        const double wall = region.half.z() - region.contact_margin;
        local.translation().z() += std::uniform_real_distribution<double>(-wall, wall)(rng_);
        if (!contactGeometry(region, object, local)) { ok = false; break; }
        if (attempt > 0) state.setToRandomPositions(groups_.at(region.robot));
        const Eigen::Isometry3d target = object * local;
        if (!state.setFromIK(groups_.at(region.robot), target, region.robot + "_rocobrick_tcp", .03))
          { ok = false; break; }
        state.update();
        const auto actual = state.getGlobalLinkTransform(region.robot + "_rocobrick_tcp");
        if ((actual.translation() - target.translation()).norm() > .001 ||
            Eigen::AngleAxisd(actual.linear().transpose() * target.linear()).angle() > .015 ||
            !contactGeometry(region, object, object.inverse() * actual))
          { ok = false; break; }
      }
      if (!ok || !state.satisfiesBounds(jmg_) || !constraints_->decide(state).satisfied ||
          scene_->isStateColliding(state, jmg_->getName(), false)) continue;
      if (group_state_validity_callback_)
      {
        std::vector<double> positions;
        state.copyJointGroupPositions(jmg_, positions);
        if (!group_state_validity_callback_(&state, jmg_, positions.data())) continue;
      }
      return true;
    }
    return false;
  }

  bool project(moveit::core::RobotState& state, unsigned int attempts) override { return sample(state, state, attempts); }
  const std::string& getName() const override
  {
    static const std::string name = "RoCoDualArmGraspRegionSampler";
    return name;
  }

private:
  std::vector<Region> regions_;
  std::vector<constraint_samplers::ConstraintSamplerPtr> peers_;
  std::map<std::string, const moveit::core::JointModelGroup*> groups_;
  std::shared_ptr<kinematic_constraints::KinematicConstraintSet> constraints_;
  std::mt19937 rng_;
  std::mutex mutex_;
};
}  // namespace rocobrick
