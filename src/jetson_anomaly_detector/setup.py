from setuptools import setup

package_name = 'jetson_anomaly_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Filip Sijak',
    maintainer_email='sijakf3@gmail.com',
    description='Jetson YOLO anomaly client over Raspberry Pi rosbridge WebSocket.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'jetson_yolo_rosbridge_client = jetson_anomaly_detector.jetson_yolo_rosbridge_client:main',
        ],
    },
)
