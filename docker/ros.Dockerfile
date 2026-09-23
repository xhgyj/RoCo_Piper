ARG ROS_BASE_IMAGE=osrf/ros@sha256:c3eb3a8576b8b4fee5988244135855a1a2146df93ee1523878cebdc0fb1be9a6
FROM ${ROS_BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=humble \
    ROS_LOCALHOST_ONLY=0 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-controller-manager \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-moveit \
    ros-humble-moveit-configs-utils \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-topic-based-ros2-control \
    ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ros_ws

COPY docker/ros-entrypoint.sh /usr/local/bin/ros-entrypoint
RUN chmod 0555 /usr/local/bin/ros-entrypoint

ENTRYPOINT ["/usr/local/bin/ros-entrypoint"]
CMD ["sleep", "infinity"]
