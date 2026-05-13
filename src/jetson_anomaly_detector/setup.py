from setuptools import setup

package_name = 'jetson_anomaly_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/anomaly_detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Filip Sijak',
    maintainer_email='sijakf3@gmail.com',
    description='ROS 2 companion node for camera-based anomaly events on Jetson.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'anomaly_detector = jetson_anomaly_detector.anomaly_detector_node:main',
        ],
    },
)
