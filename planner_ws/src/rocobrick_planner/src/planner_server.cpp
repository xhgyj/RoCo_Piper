#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include "planner.grpc.pb.h"

namespace api = rocobrick::planner::v1;

class PlannerService final : public api::Planner::Service {
 public:
  explicit PlannerService(const rclcpp::Node::SharedPtr& node) : node_(node) {
    robot_model_loader::RobotModelLoader loader(node_, "robot_description");
    collision_scene_ = std::make_shared<planning_scene::PlanningScene>(loader.getModel());
    baseline_acm_ = collision_scene_->getAllowedCollisionMatrix();
    pipeline_ = std::make_shared<planning_pipeline::PlanningPipeline>(loader.getModel(), node_, "ompl");
    pipeline_->displayComputedMotionPlans(false);
    pipeline_->publishReceivedRequests(false);
    pipeline_->checkSolutionPaths(true);
  }

  grpc::Status Health(grpc::ServerContext*, const api::HealthRequest*, api::HealthResponse* response) override {
    response->set_ready(true);
    response->set_model_name("rocobrick_dual_piper_l");
    response->set_trajectory_execution_enabled(false);
    return grpc::Status::OK;
  }

  grpc::Status Plan(grpc::ServerContext*, const api::PlanRequest* request, api::PlanResponse* response) override {
    std::lock_guard<std::mutex> guard(scene_mutex_);
    response->set_request_id(request->request_id());
    *response->mutable_revisions() = request->revisions();
    if (!Valid(*request)) {
      response->set_status(api::PlanResponse::INVALID_REQUEST);
      return grpc::Status::OK;
    }
    response->set_start_digest(Digest(request->start().SerializeAsString()));
    response->set_goal_digest(Digest(request->target().SerializeAsString()));
    std::string scene_bytes;
    for (const auto& object : request->objects()) scene_bytes += object.SerializeAsString();
    for (const auto& allowance : request->allowed_contacts()) scene_bytes += allowance.SerializeAsString();
    response->set_scene_digest(Digest(scene_bytes));
    const std::string group_name = request->robot_id() + "_arm";
    try {
      moveit::core::RobotState start(collision_scene_->getRobotModel());
      start.setToDefaultValues();
      SetMeasuredRobotState(*request, start);
      ReplaceScene(*request);
      collision_scene_->setCurrentState(start);
      geometry_msgs::msg::PoseStamped target;
      target.header.frame_id = request->target().frame_id();
      target.pose.position.x = request->target().position_m(0);
      target.pose.position.y = request->target().position_m(1);
      target.pose.position.z = request->target().position_m(2);
      target.pose.orientation.x = request->target().orientation_xyzw(0);
      target.pose.orientation.y = request->target().orientation_xyzw(1);
      target.pose.orientation.z = request->target().orientation_xyzw(2);
      target.pose.orientation.w = request->target().orientation_xyzw(3);
      planning_interface::MotionPlanRequest plan_request;
      plan_request.group_name = group_name;
      plan_request.allowed_planning_time = 30.0;
      plan_request.max_velocity_scaling_factor = 0.01;
      plan_request.max_acceleration_scaling_factor = 0.01;
      moveit::core::robotStateToRobotStateMsg(start, plan_request.start_state);
      plan_request.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(
          request->robot_id() + "_tcp_link", target, 0.001, 0.01));
      planning_interface::MotionPlanResponse plan_response;
      if (!pipeline_->generatePlan(collision_scene_, plan_request, plan_response) ||
          plan_response.error_code_.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
          !plan_response.trajectory_) {
        response->set_status(api::PlanResponse::PLANNING_FAILED);
        response->add_collision_report("MoveIt did not find an admitted path");
        ExplainTargetFailure(start, group_name, target.pose, response);
        return grpc::Status::OK;
      }
      moveit_msgs::msg::RobotTrajectory planned;
      plan_response.trajectory_->getRobotTrajectoryMsg(planned);
      if (!ValidateOtherMotion(planned.joint_trajectory, request->other_robot_remaining_motion(), *request)) {
        response->set_status(api::PlanResponse::COLLISION_BLOCKED);
        response->add_collision_report("candidate conflicts with the supplied other-arm remaining motion");
        return grpc::Status::OK;
      }
      auto* trajectory = response->mutable_trajectory();
      for (const auto& name : planned.joint_trajectory.joint_names) trajectory->add_joint_names(name);
      for (const auto& point : planned.joint_trajectory.points) {
        auto* output = trajectory->add_points();
        for (double position : point.positions) output->add_positions(position);
        output->set_time_from_start_s(rclcpp::Duration(point.time_from_start).seconds());
      }
      response->set_status(api::PlanResponse::SUCCESS);
      response->add_required_resources(request->robot_id());
      return grpc::Status::OK;
    } catch (const std::exception& error) {
      response->set_status(api::PlanResponse::PLANNING_FAILED);
      response->add_collision_report(error.what());
      return grpc::Status::OK;
    }
  }

 private:
  bool Valid(const api::PlanRequest& request) const {
    if (request.robot_id() != "piper_0" && request.robot_id() != "piper_1") return false;
    if (request.start().names_size() == 0 || request.start().names_size() != request.start().positions_size()) return false;
    if (request.target().position_m_size() != 3 || request.target().orientation_xyzw_size() != 4) return false;
    if (request.robots_size() != 2) return false;
    std::unordered_set<std::string> robot_ids;
    for (const auto& robot : request.robots()) {
      if (!robot_ids.insert(robot.robot_id()).second) return false;
      if (robot.joints().names_size() != 6 || robot.joints().positions_size() != 6) return false;
    }
    if (!robot_ids.count("piper_0") || !robot_ids.count("piper_1")) return false;
    bool matching_start = false;
    for (const auto& robot : request.robots()) {
      if (robot.robot_id() != request.robot_id()) continue;
      matching_start = robot.joints().SerializeAsString() == request.start().SerializeAsString();
    }
    if (!matching_start) return false;
    for (const auto& object : request.objects()) {
      if (object.object_id().empty() || object.pose().position_m_size() != 3 ||
          object.pose().orientation_xyzw_size() != 4) return false;
      const int dimensions = object.dimensions_m_size();
      if ((object.shape() == api::CollisionObject::BOX && dimensions != 3) ||
          (object.shape() == api::CollisionObject::SPHERE && dimensions != 1) ||
          (object.shape() == api::CollisionObject::CYLINDER && dimensions != 2)) return false;
      for (double dimension : object.dimensions_m()) if (!(dimension > 0.0)) return false;
    }
    double norm = 0.0;
    for (double value : request.target().orientation_xyzw()) norm += value * value;
    return std::abs(norm - 1.0) < 1e-4;
  }

  static geometry_msgs::msg::Pose ToPose(const api::Pose& input) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = input.position_m(0);
    pose.position.y = input.position_m(1);
    pose.position.z = input.position_m(2);
    pose.orientation.x = input.orientation_xyzw(0);
    pose.orientation.y = input.orientation_xyzw(1);
    pose.orientation.z = input.orientation_xyzw(2);
    pose.orientation.w = input.orientation_xyzw(3);
    return pose;
  }

  static moveit_msgs::msg::CollisionObject ToCollisionObject(const api::CollisionObject& object) {
    moveit_msgs::msg::CollisionObject message;
    message.header.frame_id = object.pose().frame_id();
    message.id = object.object_id();
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = object.shape() == api::CollisionObject::SPHERE
                         ? shape_msgs::msg::SolidPrimitive::SPHERE
                         : object.shape() == api::CollisionObject::CYLINDER
                               ? shape_msgs::msg::SolidPrimitive::CYLINDER
                               : shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions.assign(object.dimensions_m().begin(), object.dimensions_m().end());
    message.primitives.push_back(primitive);
    message.primitive_poses.push_back(ToPose(object.pose()));
    message.operation = moveit_msgs::msg::CollisionObject::ADD;
    return message;
  }

  void SetMeasuredRobotState(const api::PlanRequest& request, moveit::core::RobotState& state) const {
    for (const auto& robot : request.robots()) {
      state.setVariablePositions(
          std::vector<std::string>(robot.joints().names().begin(), robot.joints().names().end()),
          std::vector<double>(robot.joints().positions().begin(), robot.joints().positions().end()));
    }
    state.update();
  }

  void ReplaceScene(const api::PlanRequest& request) {
    const auto known = scene_.getKnownObjectNames();
    if (!known.empty()) scene_.removeCollisionObjects(known);
    for (const auto& id : attached_ids_) {
      moveit_msgs::msg::AttachedCollisionObject removal;
      removal.object.id = id;
      removal.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      scene_.applyAttachedCollisionObject(removal);
    }
    attached_ids_.clear();
    collision_scene_->getWorldNonConst()->clearObjects();
    collision_scene_->getAllowedCollisionMatrixNonConst() = baseline_acm_;

    std::vector<moveit_msgs::msg::CollisionObject> world;
    for (const auto& object : request.objects()) {
      auto message = ToCollisionObject(object);
      if (object.attached_link().empty()) {
        world.push_back(message);
      } else {
        moveit_msgs::msg::AttachedCollisionObject attached;
        attached.link_name = object.attached_link();
        attached.object = message;
        attached.touch_links.push_back(object.attached_link());
        for (const auto& allowance : request.allowed_contacts()) {
          if (allowance.first() == object.object_id() && allowance.second() != object.attached_link()) {
            attached.touch_links.push_back(allowance.second());
          } else if (allowance.second() == object.object_id() && allowance.first() != object.attached_link()) {
            attached.touch_links.push_back(allowance.first());
          }
        }
        scene_.applyAttachedCollisionObject(attached);
        collision_scene_->processAttachedCollisionObjectMsg(attached);
        attached_ids_.insert(object.object_id());
      }
    }
    scene_.applyCollisionObjects(world);
    for (const auto& message : world) collision_scene_->processCollisionObjectMsg(message);

    auto& acm = collision_scene_->getAllowedCollisionMatrixNonConst();
    for (const auto& allowance : request.allowed_contacts()) {
      acm.setEntry(allowance.first(), allowance.second(), true);
    }
    moveit_msgs::msg::PlanningScene diff;
    diff.is_diff = true;
    collision_scene_->getAllowedCollisionMatrix().getMessage(diff.allowed_collision_matrix);
    scene_.applyPlanningScene(diff);
  }

  bool ValidateOtherMotion(
      const trajectory_msgs::msg::JointTrajectory& candidate,
      const api::Trajectory& other,
      const api::PlanRequest& request) const {
    if (other.points_size() == 0) return true;
    if (candidate.points.empty()) return false;
    for (const auto& point : other.points()) {
      if (point.positions_size() != other.joint_names_size()) return false;
    }
    moveit::core::RobotState state(collision_scene_->getRobotModel());
    state.setToDefaultValues();
    SetMeasuredRobotState(request, state);
    const double candidate_end = rclcpp::Duration(candidate.points.back().time_from_start).seconds();
    const double other_end = other.points(other.points_size() - 1).time_from_start_s();
    const int samples = std::max<int>(candidate.points.size(), other.points_size());
    for (int sample = 0; sample < samples; ++sample) {
      const double ratio = samples == 1 ? 1.0 : static_cast<double>(sample) / (samples - 1);
      SetInterpolated(candidate.joint_names, candidate.points, ratio * candidate_end, state);
      SetInterpolated(other.joint_names(), other.points(), ratio * other_end, state);
      state.update();
      collision_detection::CollisionRequest collision_request;
      collision_request.contacts = true;
      collision_request.max_contacts = 1;
      collision_detection::CollisionResult collision_result;
      collision_scene_->checkCollision(collision_request, collision_result, state);
      if (collision_result.collision) return false;
    }
    return true;
  }

  void ExplainTargetFailure(
      const moveit::core::RobotState& start, const std::string& group_name,
      const geometry_msgs::msg::Pose& target, api::PlanResponse* response) const {
    moveit::core::RobotState goal(start);
    const auto* group = goal.getJointModelGroup(group_name);
    const auto robot_id = group_name.substr(0, group_name.size() - 4);
    const auto tip_link = robot_id + "_tcp_link";
    if (!group || !goal.setFromIK(group, target, tip_link, 1.0)) {
      response->add_collision_report("target inverse kinematics has no solution");
      return;
    }
    goal.update();
    collision_detection::CollisionRequest request;
    request.contacts = true;
    request.max_contacts = 100;
    request.max_contacts_per_pair = 1;
    collision_detection::CollisionResult result;
    collision_scene_->checkCollision(request, result, goal);
    if (!result.collision) {
      response->add_collision_report("target IK is collision-free; path search failed");
      return;
    }
    for (const auto& [pair, contacts] : result.contacts) {
      response->add_collision_report("target collision: " + pair.first + " <-> " + pair.second);
    }
  }

  template <typename Names, typename Points>
  void SetInterpolated(const Names& names, const Points& points, double time, moveit::core::RobotState& state) const {
    if (points.size() == 0) return;
    std::size_t upper = 0;
    while (upper + 1 < static_cast<std::size_t>(points.size()) && PointTime(points[upper]) < time) ++upper;
    const std::size_t lower = upper == 0 ? 0 : upper - 1;
    const double t0 = PointTime(points[lower]);
    const double t1 = PointTime(points[upper]);
    const double alpha = t1 <= t0 ? 0.0 : std::clamp((time - t0) / (t1 - t0), 0.0, 1.0);
    for (std::size_t index = 0; index < static_cast<std::size_t>(names.size()); ++index) {
      const double value = PointPosition(points[lower], index) + alpha * (PointPosition(points[upper], index) - PointPosition(points[lower], index));
      state.setVariablePosition(names[index], value);
    }
  }

  static double PointTime(const trajectory_msgs::msg::JointTrajectoryPoint& point) {
    return rclcpp::Duration(point.time_from_start).seconds();
  }
  static double PointTime(const api::TrajectoryPoint& point) { return point.time_from_start_s(); }
  static double PointPosition(const trajectory_msgs::msg::JointTrajectoryPoint& point, std::size_t index) { return point.positions.at(index); }
  static double PointPosition(const api::TrajectoryPoint& point, std::size_t index) { return point.positions(static_cast<int>(index)); }

  static std::string Digest(const std::string& bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : bytes) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    std::ostringstream output;
    output << std::hex << hash;
    return output.str();
  }

  rclcpp::Node::SharedPtr node_;
  moveit::planning_interface::PlanningSceneInterface scene_;
  planning_scene::PlanningScenePtr collision_scene_;
  planning_pipeline::PlanningPipelinePtr pipeline_;
  collision_detection::AllowedCollisionMatrix baseline_acm_;
  std::unordered_set<std::string> attached_ids_;
  std::mutex scene_mutex_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("rocobrick_planner_server", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  const std::string address = node->get_parameter("listen_address").as_string();
  PlannerService service(node);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  if (!server) {
    RCLCPP_FATAL(node->get_logger(), "failed to start gRPC planner on %s", address.c_str());
    return 2;
  }
  std::thread grpc_thread([&server]() { server->Wait(); });
  rclcpp::spin(node);
  server->Shutdown();
  grpc_thread.join();
  rclcpp::shutdown();
  return 0;
}
