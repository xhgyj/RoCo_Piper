// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <moveit/constraint_samplers/constraint_sampler_manager.h>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <mutex>
#include <random>
#include <sstream>

namespace rocobrick
{
// This is MoveIt's goal sampler, not a host IK/trajectory planner. OMPL still
// owns the combined path and collision/constraint admission. Sample the giver
// region FIRST, then compute receiver geometry from that sampled carried brick.
class HandoverRegionSampler final : public constraint_samplers::ConstraintSampler
{
public:
  HandoverRegionSampler(const planning_scene::PlanningSceneConstPtr& scene, const std::string& group)
    : ConstraintSampler(scene, group), rng_(std::random_device{}())
  {
  }

  bool configure(const moveit_msgs::msg::Constraints& constraints) override
  {
    is_valid_ = false;
    std::istringstream stream(constraints.name);
    std::string token, donor, binding;
    if (!std::getline(stream, token, ';') || token != "rocobrick-handover-region/v1" ||
        !std::getline(stream, donor, ';') || (donor != "piper_0" && donor != "piper_1") ||
        !std::getline(stream, binding, ';') || binding.size() != 64)
      return false;
    std::array<double, 17> values{};
    for (auto& value : values)
    {
      if (!std::getline(stream, token, ';'))
        return false;
      try
      {
        std::size_t used = 0;
        value = std::stod(token, &used);
        if (used != token.size() || !std::isfinite(value))
          return false;
      }
      catch (const std::exception&)
      {
        return false;
      }
    }
    if (std::getline(stream, token, ';'))
      return false;
    Eigen::Quaterniond quaternion(values[6], values[3], values[4], values[5]);
    if (std::abs(quaternion.norm() - 1.) > 1e-6)
      return false;
    object_in_giver_ = Eigen::Isometry3d::Identity();
    object_in_giver_.linear() = quaternion.toRotationMatrix();
    object_in_giver_.translation() = Eigen::Vector3d(values[0], values[1], values[2]);
    center_ = Eigen::Vector3d(values[7], values[8], values[9]);
    extents_ = Eigen::Vector3d(values[10], values[11], values[12]);
    precontact_ = values[13];
    clearance_ = values[14];
    lateral_ = values[15];
    depth_ = values[16];
    if ((extents_.array() <= 0).any() || precontact_ <= 0 || precontact_ > .08 ||
        clearance_ <= 0 || clearance_ > .01 || lateral_ <= 0 || lateral_ > .004 ||
        depth_ <= 0 || depth_ > .004)
      return false;
    donor_tip_ = donor + "_rocobrick_tcp";
    receiver_tip_ = (donor == "piper_0" ? "piper_1" : "piper_0") + std::string("_rocobrick_tcp");
    const auto model = scene_->getRobotModel();
    receiver_group_ = model->getJointModelGroup((donor == "piper_0" ? "piper_1" : "piper_0") + std::string("_arm_tcp"));
    moveit_msgs::msg::Constraints giver_constraints;
    for (const auto& constraint : constraints.position_constraints)
      if (constraint.link_name == donor_tip_)
        giver_constraints.position_constraints.push_back(constraint);
    for (const auto& constraint : constraints.orientation_constraints)
      if (constraint.link_name == donor_tip_)
        giver_constraints.orientation_constraints.push_back(constraint);
    if (giver_constraints.position_constraints.size() != 1 || giver_constraints.orientation_constraints.size() != 1 ||
        !receiver_group_ || !receiver_group_->getSolverInstance())
      return false;
    giver_sampler_ = constraint_samplers::ConstraintSamplerManager::selectDefaultSampler(
        scene_, donor + "_arm_tcp", giver_constraints);
    is_valid_ = jmg_ && giver_sampler_ && giver_sampler_->isValid();
    return is_valid_;
  }

  using constraint_samplers::ConstraintSampler::sample;
  bool sample(moveit::core::RobotState& state, const moveit::core::RobotState& reference,
              unsigned int attempts) override
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_valid_)
      return false;
    moveit::core::RobotState stable(reference);
    stable.update();
    for (unsigned int i = 0; i < attempts; ++i)
    {
      state = stable;
      if (!giver_sampler_->sample(state, stable, 1))
        continue;
      state.update();
      const auto object = state.getGlobalLinkTransform(donor_tip_) * object_in_giver_;
      // Side pinch on either brick-local X or Y. Continuous angle about that
      // closing axis: sideways and oblique are legal, no world-up restriction.
      // The palm remains behind the stud-face half-space (local +Z exposed).
      const double yaw = std::uniform_int_distribution<int>(0, 3)(rng_) * M_PI / 2.;
      const double pitch = std::uniform_real_distribution<double>(-M_PI / 2., M_PI / 2.)(rng_);
      const Eigen::Matrix3d rotation =
          (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())).toRotationMatrix();
      const double lateral_limit = std::min(lateral_, .25 * std::min(extents_.x(), extents_.y()));
      Eigen::Isometry3d receiver = Eigen::Isometry3d::Identity();
      receiver.linear() = rotation;
      receiver.translation() = center_ + rotation.col(0) *
          std::uniform_real_distribution<double>(-lateral_limit, lateral_limit)(rng_) -
          rotation.col(2) * (precontact_ + clearance_ +
          std::uniform_real_distribution<double>(0., depth_)(rng_));
      const Eigen::Isometry3d target = object * receiver;
      if (!state.setFromIK(receiver_group_, target, receiver_tip_, .02))
        continue;
      state.update();
      if (!state.satisfiesBounds(jmg_))
        continue;
      const auto actual = state.getGlobalLinkTransform(receiver_tip_);
      if ((actual.translation() - target.translation()).norm() > .001 ||
          Eigen::AngleAxisd(actual.linear().transpose() * target.linear()).angle() > .015)
        continue;
      if (group_state_validity_callback_)
      {
        std::vector<double> positions;
        state.copyJointGroupPositions(jmg_, positions);
        if (!group_state_validity_callback_(&state, jmg_, positions.data()))
          continue;
      }
      return true;
    }
    return false;
  }

  bool project(moveit::core::RobotState& state, unsigned int attempts) override
  {
    return sample(state, state, attempts);
  }

  const std::string& getName() const override
  {
    static const std::string name = "RoCoCoupledHandoverRegionSampler";
    return name;
  }

private:
  constraint_samplers::ConstraintSamplerPtr giver_sampler_;
  const moveit::core::JointModelGroup* receiver_group_ = nullptr;
  std::string donor_tip_, receiver_tip_;
  Eigen::Isometry3d object_in_giver_;
  Eigen::Vector3d center_, extents_;
  double precontact_ = 0, clearance_ = 0, lateral_ = 0, depth_ = 0;
  std::mt19937 rng_;
  std::mutex mutex_;
};
}  // namespace rocobrick
