cmake_minimum_required(VERSION 3.16)
project(rocobrick_planner)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPCPP REQUIRED grpc++)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/planner.proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})
add_custom_command(
  OUTPUT ${GENERATED_DIR}/planner.pb.cc ${GENERATED_DIR}/planner.pb.h
         ${GENERATED_DIR}/planner.grpc.pb.cc ${GENERATED_DIR}/planner.grpc.pb.h
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
  ARGS --cpp_out=${GENERATED_DIR} --grpc_out=${GENERATED_DIR}
       --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
       -I ${CMAKE_CURRENT_SOURCE_DIR}/proto ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
)
add_library(planner_proto
  ${GENERATED_DIR}/planner.pb.cc
  ${GENERATED_DIR}/planner.grpc.pb.cc
)
target_include_directories(planner_proto PUBLIC ${GENERATED_DIR} ${Protobuf_INCLUDE_DIRS} ${GRPCPP_INCLUDE_DIRS})
target_link_directories(planner_proto PUBLIC ${GRPCPP_LIBRARY_DIRS})
target_link_libraries(planner_proto PUBLIC ${Protobuf_LIBRARIES} ${GRPCPP_LIBRARIES})

add_executable(planner_server src/planner_server.cpp)
target_compile_features(planner_server PRIVATE cxx_std_17)
target_link_libraries(planner_server planner_proto)
ament_target_dependencies(planner_server rclcpp geometry_msgs moveit_core moveit_ros_planning moveit_ros_planning_interface moveit_msgs shape_msgs)
install(TARGETS planner_server DESTINATION lib/${PROJECT_NAME})
install(FILES ${PROTO_FILE} DESTINATION share/${PROJECT_NAME}/proto)
ament_package()
