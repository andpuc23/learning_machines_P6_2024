#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /root/catkin_ws/setup.bash

# rosrun learning_machines learning_robobo_controller.py "$@"
# rosrun learning_machines task_0_run_till_wall.py "$@"
# rosrun learning_machines task_0_test_sensors.py "$@"
# rosrun learning_machines task_1_avoid_obstacles.py "$@"
# rosrun learning_machines task_2_old_code.py "$@"
rosrun learning_machines task_3.py "$@"