cmake_minimum_required(VERSION 3.22)
project(rocobrick_piper_l_moveit_config)

find_package(ament_cmake REQUIRED)

ament_package()

# Config-only package: no compiled code, just config, launch, and urdf share data.
install(
  DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/dual_joint_state_aggregator.py
  DESTINATION lib/${PROJECT_NAME}
)
