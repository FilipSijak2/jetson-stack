#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_file = LaunchConfiguration('config_file')
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value='/workspace/config/anomaly_detector.yaml',
            description='Path to anomaly detector YAML config',
        ),
        Node(
            package='jetson_anomaly_detector',
            executable='anomaly_detector',
            name='jetson_anomaly_detector',
            output='screen',
            emulate_tty=True,
            parameters=[{'config_file': config_file}],
        ),
    ])
