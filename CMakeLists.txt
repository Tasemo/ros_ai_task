cmake_minimum_required(VERSION 3.0.2)
project(ros_ai_task)

find_package(
  catkin 
  REQUIRED COMPONENTS
  rospy 
  std_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  Int32Header.msg
)

add_service_files(
  FILES
  ImagePrediction.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  message_runtime
  DEPENDS
  python-pytorch-pip
  python3-opencv
  python3-matplotlib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
  scripts/ai.py
  scripts/camera.py
  scripts/controller.py
  scripts/processor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
