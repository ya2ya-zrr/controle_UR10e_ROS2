from setuptools import find_packages, setup

package_name = 'ur10e_ik_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='localuser',
    maintainer_email='localuser@todo.todo',
    description='Contrôle IK direct pour UR10e via MuJoCo et ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_node = ur10e_ik_control.ik_node:main',
            'vision_node = ur10e_ik_control.vision_node:main',
            'planner = ur10e_ik_control.high_level_planner:main', 
            'vision_node_yolo = ur10e_ik_control.vision_node_yolo:main',
            'data_collector = ur10e_ik_control.script_collecte_data:main',
            'ball_tracker_cnn = ur10e_ik_control.ball_tracker_cnn:main',
        ],
    },
)
