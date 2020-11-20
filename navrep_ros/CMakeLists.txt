cmake_minimum_required(VERSION 2.8.3)
project(navrep_ros)

find_package(catkin_simple REQUIRED)

catkin_simple()

cs_install()
cs_install_scripts(scripts/ros_run_navrep)
