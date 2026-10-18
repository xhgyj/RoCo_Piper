// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <moveit/constraint_samplers/constraint_sampler_manager.h>
#include <moveit/kinematic_constraints/kinematic_constraint.h>
#include <moveit/collision_detection/collision_common.h>
#include <Eigen/Geometry>
#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <map>
#include <vector>
#include <random>
#include <set>
#include <sstream>
#include <string>

namespace rocobrick
{
// Diagnostic-only instrumentation. Nothing in this namespace is consulted by a
// sampling decision: the gates, thresholds, sampler choice and geometry below
// are unchanged. Every record is one line on stderr, greppable as "[pgrd]".
namespace grasp_diag
{
inline constexpr const char* kVersion = "pgrd/1";  // bump when the record layout changes
inline constexpr unsigned kMaxDetail = 3;          // full detail for the first failed attempts only
inline constexpr unsigned kMaxCollisionDumps = 3;  // bounded contact queries, never on the hot path
inline constexpr unsigned kMaxContactPairs = 4;
inline constexpr std::int64_t kSummaryPeriodNs = 1000000000;  // at most one periodic summary per second

inline std::uint64_t nextInstanceId()
{
  static std::atomic<std::uint64_t> counter{ 0 };
  return ++counter;
}

// Evaluation order of the existing gates. The first stage that rejects an
// attempt is that attempt's rejection, so one attempt can never be counted more
// than once (per-region misses are not attempt counts).
enum Stage : unsigned
{
  kAccepted = 0,
  kPeer,
  kBaseFrame,
  kGeometry,
  kIk,
  kKinematicsResidual,  // post-IK FK/geometry re-check
  kBounds,
  kConstraints,
  kCollision,
  kCallback,
  kStageCount
};

inline const char* stageName(unsigned stage)
{
  switch (stage)
  {
    case kAccepted:
      return "accepted";
    case kPeer:
      return "peer";
    case kBaseFrame:
      return "base_frame";
    case kGeometry:
      return "geometry";
    case kIk:
      return "ik";
    case kKinematicsResidual:
      return "kinematics_residual";
    case kBounds:
      return "bounds";
    case kConstraints:
      return "constraints";
    case kCollision:
      return "collision";
    case kCallback:
      return "callback";
    default:
      return "unknown";
  }
}

// Mirrors the branches of contactGeometry(); index 0 means accepted.
inline const char* geometryReasonName(int reason)
{
  switch (reason)
  {
    case 0:
      return "ok";
    case 1:
      return "closing_axis";
    case 2:
      return "longitudinal_offset";
    case 3:
      return "tangential_offset";
    case 4:
      return "height_band";
    case 5:
      return "tcp_z_axis";
    case 6:
      return "flange_inside_body";
    case 7:
      return "opening_too_narrow";
    case 8:
      return "pad_below_ground";
    default:
      return "unknown";
  }
}

struct Counters
{
  std::uint64_t calls = 0;             // sample() entries with a valid configuration
  std::uint64_t invalid_calls = 0;     // sample() entries before/after a failed configure()
  std::uint64_t attempts_budget = 0;   // sum of the `attempts` budgets handed in by the caller
  std::uint64_t attempts_run = 0;      // attempts actually evaluated
  std::uint64_t accepted = 0;
  std::uint64_t accept_attempt_sum = 0;
  std::uint64_t ik_calls = 0;
  std::uint64_t ik_fail = 0;
  std::uint64_t ik_ns = 0;
  std::uint64_t collision_dumps = 0;
  std::array<std::uint64_t, kStageCount> first{};  // per attempt, first rejecting stage
  std::array<std::uint64_t, kStageCount> hits{};   // every gate hit (may exceed attempts_run)

  void reject(unsigned stage)
  {
    if (stage >= kStageCount) return;
    ++hits[stage];
    ++first[stage];
  }

  std::string summary() const
  {
    std::uint64_t first_total = 0;
    for (const std::uint64_t value : first) first_total += value;
    std::ostringstream out;
    out << "calls=" << calls << " calls_invalid=" << invalid_calls << " attempts_budget=" << attempts_budget
        << " attempts_run=" << attempts_run << " accepted=" << accepted << " first_reject_total=" << first_total;
    if (accepted)
      out << " accepted_attempt_avg=" << static_cast<double>(accept_attempt_sum) / static_cast<double>(accepted);
    out << " ik_calls=" << ik_calls << " ik_fail=" << ik_fail << " ik_ms=" << static_cast<double>(ik_ns) / 1e6;
    if (ik_calls)
      out << " ik_ms_avg=" << static_cast<double>(ik_ns) / 1e6 / static_cast<double>(ik_calls);
    out << " collision_dumps=" << collision_dumps << " first_reject{";
    for (unsigned stage = 1; stage < kStageCount; ++stage)
      if (first[stage]) out << ' ' << stageName(stage) << '=' << first[stage];
    out << " } stage_hits{";
    for (unsigned stage = 1; stage < kStageCount; ++stage)
      if (hits[stage]) out << ' ' << stageName(stage) << '=' << hits[stage];
    out << " }";
    return out.str();
  }
};

// Per-attempt trace, filled by the existing gates in evaluation order.
struct Attempt
{
  unsigned stage = kAccepted;
  std::size_t region = 0;
  std::string robot;
  std::string bounds;  // filled for kBounds
  double face = 0, pitch = 0, flip = 0;
  int geometry = 0;
  double base_trans = 0, base_rot = 0;  // filled for kBaseFrame
  double res_trans = 0, res_rot = 0;    // filled for kKinematicsResidual
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  Eigen::Vector3d local = Eigen::Vector3d::Zero();

  void note(unsigned value)
  {
    if (stage == kAccepted) stage = value;
  }
};
}  // namespace grasp_diag

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
    : ConstraintSampler(scene, group), seed_(static_cast<std::uint32_t>(std::random_device{}())), rng_(seed_)
  {
  }

  ~GraspRegionSampler() override
  {
    try
    {
      std::lock_guard<std::mutex> guard(mutex_);
      log_line("event=final " + diag_.summary());
    }
    catch (...)
    {
      // A destructor must never throw, in particular while ROS is shutting down.
    }
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
    auto refuse = [&](const char* reason) {
      std::ostringstream note;
      note << "event=configure ok=0 reason=" << reason << " group=" << (jmg_ ? jmg_->getName() : "none")
           << " regions=" << regions_.size() << " binding_bytes=" << constraints.name.size();
      log_line(note.str());
      return false;
    };
    if (!jmg_ || !constraints.visibility_constraints.empty() || !decode(constraints.name, regions_))
      return refuse("sampler_precondition_or_decode");
    const auto model = scene_->getRobotModel();
    for (const std::string robot : {"piper_0", "piper_1"})
    {
      const auto* group = model->getJointModelGroup(robot + "_arm_tcp");
      if (!group || !model->getLinkModel(robot + "_base_link") || !model->getLinkModel(robot + "_rocobrick_tcp"))
        return refuse("missing_group_or_link");
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
          return refuse("owned_region_scoped_constraints");
      }
      else
      {
        auto sampler = constraint_samplers::ConstraintSamplerManager::selectDefaultSampler(scene_, robot + "_arm_tcp", scoped);
        if (!sampler || !sampler->isValid()) return refuse("peer_sampler_unavailable");
        peers_.push_back(sampler);
      }
    }
    constraints_ = std::make_shared<kinematic_constraints::KinematicConstraintSet>(model);
    constraints_->add(constraints, scene_->getTransforms());
    is_valid_ = true;
    std::ostringstream note;
    note << "event=configure ok=1 group=" << jmg_->getName() << " regions=" << regions_.size() << " peers=" << peers_.size()
         << " owners{";
    for (const Region& region : regions_) note << ' ' << region.robot;
    note << " } binding_bytes=" << constraints.name.size();
    log_line(note.str());
    return true;
  }

  // Contact geometry only. Ground/body/model/attached collisions are still
  // independently checked by MoveIt on the sampled state and the full path.
  // `reason` is a diagnostic out-parameter (0 == accepted); it changes neither
  // the thresholds nor the returned decision.
  static bool contactGeometry(const Region& region, const Eigen::Isometry3d& object_world,
                              const Eigen::Isometry3d& local, int* reason = nullptr)
  {
    const Eigen::Vector3d closing = local.linear().col(1);
    const unsigned axis = std::abs(closing.x()) > std::abs(closing.y()) ? 0 : 1;
    const unsigned tangent = 1 - axis;
    const Eigen::Vector3d delta = local.translation() - region.center;
    if (std::abs(closing[axis]) < std::cos(.015) || std::abs(closing.z()) > std::sin(.015))
    {
      if (reason) *reason = 1;
      return false;
    }
    if (std::abs(delta[axis]) > .001)
    {
      if (reason) *reason = 2;
      return false;
    }
    if (std::abs(delta[tangent]) > region.half[tangent] * .5 + .001)
    {
      if (reason) *reason = 3;
      return false;
    }
    if (std::abs(delta.z()) > region.half.z() - region.contact_margin + .001 || std::abs(local.linear()(2, 2)) >= std::cos(.015))
    {
      if (reason) *reason = std::abs(local.linear()(2, 2)) >= std::cos(.015) ? 5 : 4;
      return false;
    }
    const Eigen::Vector3d flange = delta - region.flange * local.linear().col(2);
    if (std::abs(flange.x()) < region.half.x() && std::abs(flange.y()) < region.half.y() && flange.z() > region.half.z())
    {
      if (reason) *reason = 6;
      return false;
    }
    const double span = 2. * closing.cwiseAbs().dot(region.half);
    if (region.opening < span + region.contact_margin)
    {
      if (reason) *reason = 7;
      return false;
    }
    for (const auto& pad : region.pads)
      if ((object_world * local * pad).z() < region.ground + region.floor_margin - .001)
      {
        if (reason) *reason = 8;
        return false;
      }
    return true;
  }

  using constraint_samplers::ConstraintSampler::sample;
  bool sample(moveit::core::RobotState& state, const moveit::core::RobotState& reference, unsigned int attempts) override
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_valid_)
    {
      ++diag_.invalid_calls;
      maybe_log_summary();
      return false;
    }
    ++diag_.calls;
    diag_.attempts_budget += attempts;
    if (attempts != last_budget_)
    {
      last_budget_ = attempts;
      std::ostringstream note;
      note << "event=budget attempts=" << attempts << " (requested by the caller; recorded on change)";
      log_line(note.str());
    }
    moveit::core::RobotState stable(reference);
    stable.update();
    if (diag_.calls == 1)
      log_probe(stable);
    for (unsigned attempt = 0; attempt < attempts; ++attempt)
    {
      ++diag_.attempts_run;
      grasp_diag::Attempt trace;
      state = stable;
      for (const auto& peer : peers_)
      {
        moveit::core::RobotState clean(state);
        clean.update();
        if (!peer->sample(state, clean, 1))
        {
          trace.note(grasp_diag::kPeer);
          break;
        }
        state.update();
      }
      for (std::size_t index = 0; index < regions_.size() && trace.stage == grasp_diag::kAccepted; ++index)
      {
        const Region& region = regions_[index];
        trace.region = index;
        trace.robot = region.robot;
        const auto base = stable.getGlobalLinkTransform(region.robot + "_base_link");
        trace.base_trans = (base.translation() - region.base_world.translation()).norm();
        trace.base_rot = Eigen::AngleAxisd(base.linear().transpose() * region.base_world.linear()).angle();
        if (trace.base_trans > .001 || trace.base_rot > .015)
        {
          trace.note(grasp_diag::kBaseFrame);
          break;
        }
        const auto object = base * region.object_base;
        const unsigned face = std::uniform_int_distribution<unsigned>(0, 3)(rng_);
        const double pitch = std::uniform_real_distribution<double>(0., M_PI)(rng_);
        const double flip = std::uniform_int_distribution<unsigned>(0, 1)(rng_) * M_PI;
        trace.face = face;
        trace.pitch = pitch;
        trace.flip = flip;
        Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
        local.linear() = (Eigen::AngleAxisd(face * M_PI / 2., Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(flip, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        const unsigned tangent = face % 2 == 0 ? 0 : 1;
        local.translation() = region.center;
        local.translation()[tangent] += std::uniform_real_distribution<double>(-region.half[tangent] * .5, region.half[tangent] * .5)(rng_);
        const double wall = region.half.z() - region.contact_margin;
        local.translation().z() += std::uniform_real_distribution<double>(-wall, wall)(rng_);
        trace.local = local.translation();
        trace.target = (object * local).translation();
        int geometry = 0;
        if (!contactGeometry(region, object, local, &geometry))
        {
          trace.geometry = geometry;
          trace.note(grasp_diag::kGeometry);
          break;
        }
        if (attempt > 0) state.setToRandomPositions(groups_.at(region.robot));
        const Eigen::Isometry3d target = object * local;
        trace.target = target.translation();
        const auto ik_start = std::chrono::steady_clock::now();
        const bool ik_ok = state.setFromIK(groups_.at(region.robot), target, region.robot + "_rocobrick_tcp", .03);
        diag_.ik_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - ik_start).count();
        ++diag_.ik_calls;
        if (!ik_ok)
        {
          ++diag_.ik_fail;
          trace.note(grasp_diag::kIk);
          break;
        }
        state.update();
        const auto actual = state.getGlobalLinkTransform(region.robot + "_rocobrick_tcp");
        trace.res_trans = (actual.translation() - target.translation()).norm();
        trace.res_rot = Eigen::AngleAxisd(actual.linear().transpose() * target.linear()).angle();
        int residual_geometry = 0;
        if (trace.res_trans > .001 || trace.res_rot > .015 ||
            !contactGeometry(region, object, object.inverse() * actual, &residual_geometry))
        {
          trace.geometry = residual_geometry;
          trace.note(grasp_diag::kKinematicsResidual);
          break;
        }
      }
      if (trace.stage == grasp_diag::kAccepted && !state.satisfiesBounds(jmg_))
      {
        trace.note(grasp_diag::kBounds);
        trace.bounds = describeBounds(state);
      }
      if (trace.stage == grasp_diag::kAccepted && !constraints_->decide(state).satisfied)
        trace.note(grasp_diag::kConstraints);
      if (trace.stage == grasp_diag::kAccepted && scene_->isStateColliding(state, jmg_->getName(), false))
      {
        trace.note(grasp_diag::kCollision);
        log_collision_pairs(state);
      }
      if (trace.stage == grasp_diag::kAccepted && group_state_validity_callback_)
      {
        std::vector<double> positions;
        state.copyJointGroupPositions(jmg_, positions);
        if (!group_state_validity_callback_(&state, jmg_, positions.data()))
          trace.note(grasp_diag::kCallback);
      }
      if (trace.stage == grasp_diag::kAccepted)
      {
        ++diag_.accepted;
        diag_.accept_attempt_sum += attempt + 1;
        std::ostringstream note;
        note << "event=accepted attempt=" << attempt << " regions=" << regions_.size() << " " << diag_.summary();
        log_line(note.str());
        maybe_log_summary();
        return true;
      }
      diag_.reject(trace.stage);
      log_attempt_detail(attempt, trace);
      maybe_log_summary();
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
  void log_line(const std::string& message) const
  {
    std::fprintf(stderr, "[pgrd] ver=%s inst=%llu seed=0x%08x %s\n", grasp_diag::kVersion,
                 static_cast<unsigned long long>(instance_id_), static_cast<unsigned>(seed_), message.c_str());
    std::fflush(stderr);
  }

  void maybe_log_summary()
  {
    const std::int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
    if (last_summary_ns_ != 0 && now - last_summary_ns_ < grasp_diag::kSummaryPeriodNs) return;
    last_summary_ns_ = now;
    log_line("event=summary " + diag_.summary());
  }

  // One-off snapshot of the reference state the caller handed in, plus the
  // object-frame anchors: enough to tell "sampling never succeeds" apart from
  // "the reference frame disagrees with the region".
  void log_probe(const moveit::core::RobotState& stable) const
  {
    std::ostringstream out;
    out.precision(9);
    out << "event=probe group=" << jmg_->getName() << " initial_budget=" << last_budget_ << " vars{";
    for (const std::string& name : jmg_->getVariableNames())
      out << ' ' << name << '=' << stable.getVariablePosition(name);
    out << " }";
    for (const Region& region : regions_)
    {
      const auto base = stable.getGlobalLinkTransform(region.robot + "_base_link");
      out << ' ' << region.robot << "_frame{trans_err=" << (base.translation() - region.base_world.translation()).norm()
          << " rot_err_rad=" << Eigen::AngleAxisd(base.linear().transpose() * region.base_world.linear()).angle()
          << " expected=(" << region.base_world.translation().x() << ',' << region.base_world.translation().y() << ','
          << region.base_world.translation().z() << ")}";
    }
    log_line(out.str());
  }

  void log_attempt_detail(unsigned attempt, const grasp_diag::Attempt& trace)
  {
    if (detail_used_ >= grasp_diag::kMaxDetail) return;
    ++detail_used_;
    std::ostringstream out;
    out << "event=reject_detail#" << detail_used_ << " attempt=" << attempt << " stage=" << grasp_diag::stageName(trace.stage)
        << " region=" << trace.region << " robot=" << trace.robot << " face=" << trace.face
        << " pitch_deg=" << trace.pitch * 180. / M_PI << " flip_rad=" << trace.flip;
    if (trace.geometry) out << " geometry_reason=" << grasp_diag::geometryReasonName(trace.geometry);
    if (trace.stage == grasp_diag::kBaseFrame)
      out << " base_trans_err=" << trace.base_trans << " base_rot_err_rad=" << trace.base_rot;
    if (trace.stage == grasp_diag::kKinematicsResidual)
      out << " residual_trans=" << trace.res_trans << " residual_rot_rad=" << trace.res_rot;
    if (!trace.bounds.empty()) out << " bounds{" << trace.bounds << "}";
    out << " target=(" << trace.target.x() << ',' << trace.target.y() << ',' << trace.target.z() << ')'
        << " local=(" << trace.local.x() << ',' << trace.local.y() << ',' << trace.local.z() << ')';
    log_line(out.str());
  }

  // Names the violating variables and the exceedance; the decision itself was
  // already taken by satisfiesBounds() above.
  std::string describeBounds(const moveit::core::RobotState& state) const
  {
    std::ostringstream out;
    out.precision(9);
    const auto& model = *state.getRobotModel();
    for (const std::string& name : jmg_->getVariableNames())
    {
      const auto& bounds = model.getVariableBounds(name);
      if (!bounds.position_bounded_) continue;
      const double value = state.getVariablePosition(name);
      const double below = bounds.min_position_ - value;
      const double above = value - bounds.max_position_;
      if (below <= 0. && above <= 0.) continue;
      out << ' ' << name << '=' << value << " range=[" << bounds.min_position_ << ',' << bounds.max_position_ << ']'
          << (below > 0. ? " below_min=" : " above_max=") << (below > 0. ? below : above);
    }
    return out.str();
  }

  // Bounded, diagnostic-only contact query: the collision decision above still
  // comes from isStateColliding() with the same arguments.
  void log_collision_pairs(moveit::core::RobotState& state)
  {
    if (diag_.collision_dumps >= grasp_diag::kMaxCollisionDumps) return;
    ++diag_.collision_dumps;
    collision_detection::CollisionRequest request;
    request.group_name = jmg_->getName();
    request.contacts = true;
    request.max_contacts = static_cast<std::size_t>(grasp_diag::kMaxContactPairs);
    request.max_contacts_per_pair = 2;
    collision_detection::CollisionResult result;
    scene_->checkCollision(request, result, state);
    std::ostringstream out;
    out.precision(6);
    out << "event=collision_pairs#" << diag_.collision_dumps << " contact_count=" << result.contact_count << " pairs{";
    unsigned shown = 0;
    for (const auto& pair : result.contacts)
    {
      for (const auto& contact : pair.second)
      {
        if (shown >= grasp_diag::kMaxContactPairs) break;
        ++shown;
        out << ' ' << pair.first.first << '<' << pair.first.second << " depth=" << contact.depth << " at=(" << contact.pos.x()
            << ',' << contact.pos.y() << ',' << contact.pos.z() << ')';
      }
      if (shown >= grasp_diag::kMaxContactPairs) break;
    }
    out << " }";
    log_line(out.str());
  }

  std::uint32_t seed_ = 0;
  std::mt19937 rng_;
  std::mutex mutex_;
  const std::uint64_t instance_id_ = grasp_diag::nextInstanceId();
  grasp_diag::Counters diag_;
  unsigned last_budget_ = 0;
  unsigned detail_used_ = 0;
  std::int64_t last_summary_ns_ = 0;
  std::vector<Region> regions_;
  std::vector<constraint_samplers::ConstraintSamplerPtr> peers_;
  std::map<std::string, const moveit::core::JointModelGroup*> groups_;
  std::shared_ptr<kinematic_constraints::KinematicConstraintSet> constraints_;
};
}  // namespace rocobrick
