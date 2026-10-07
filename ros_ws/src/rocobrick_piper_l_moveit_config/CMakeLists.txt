cmake_minimum_required(VERSION 3.22)
project(rocobrick_piper_l_moveit_config)

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)

add_library(rocobrick_clean_reference_sampler SHARED src/clean_reference_sampler_allocator.cpp)
target_compile_features(rocobrick_clean_reference_sampler PUBLIC cxx_std_17)
target_include_directories(rocobrick_clean_reference_sampler PRIVATE include)
ament_target_dependencies(rocobrick_clean_reference_sampler moveit_core pluginlib)
pluginlib_export_plugin_description_file(moveit_core constraint_sampler_plugins.xml)
install(TARGETS rocobrick_clean_reference_sampler LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(
  DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/dual_joint_state_aggregator.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_clean_reference_union_sampler test/test_clean_reference_union_sampler.cpp)
  target_include_directories(test_clean_reference_union_sampler PRIVATE include)
  ament_target_dependencies(test_clean_reference_union_sampler moveit_core)
endif()

ament_package()
