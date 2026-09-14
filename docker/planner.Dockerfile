FROM osrf/ros:humble-desktop-jammy

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LC_NUMERIC=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    grpc-proto libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    python3-colcon-common-extensions \
    ros-humble-moveit \
    ros-humble-moveit-configs-utils \
    ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/planner_ws
COPY planner_ws/src ./src
COPY planner_ws/UPSTREAM.json ./UPSTREAM.json

RUN source /opt/ros/humble/setup.bash && \
    colcon build --merge-install --packages-select agx_arm_description && \
    source install/setup.bash && \
    xacro src/agx_arm_description/agx_arm_urdf/piper_l/urdf/piper_l_with_gripper_description.xacro \
      > /tmp/piper_l_gripper.urdf && \
    python3 src/rocobrick_moveit_config/scripts/generate_dual_model.py \
      --single /tmp/piper_l_gripper.urdf \
      --urdf src/rocobrick_moveit_config/config/dual_piper.urdf \
      --srdf src/rocobrick_moveit_config/config/dual_piper.srdf && \
    colcon build --merge-install

COPY docker/planner-entrypoint.sh /planner-entrypoint.sh
RUN chmod 0555 /planner-entrypoint.sh
EXPOSE 50051
ENTRYPOINT ["/planner-entrypoint.sh"]
