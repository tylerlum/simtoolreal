import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    pkg_path = get_package_share_directory("wuji_hand_description")
    hand = LaunchConfiguration("hand").perform(context)
    urdf_file = os.path.join(pkg_path, "urdf", f"{hand}-ros.urdf")

    use_sim_time = LaunchConfiguration("use_sim_time")

    with open(urdf_file, "r") as infp:
        robot_desc = infp.read()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "robot_description": robot_desc}],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(pkg_path, "rviz", f"{hand}.rviz")],
    )

    return [
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hand",
                default_value="left",
                choices=["left", "right"],
                description="Hand type (left or right)",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("gui", default_value="true"),
            OpaqueFunction(function=launch_setup),
        ]
    )
