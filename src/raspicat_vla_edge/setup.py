from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'raspicat_vla_edge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'Pillow',
        'grpcio>=1.50',
    ],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='nop@example.com',
    description='Edge ROS2 nodes for VLA navigation.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_edge_node = raspicat_vla_edge.edge_node:main',
            'path_follower_node = raspicat_vla_edge.path_follower_node:main',
        ],
    },
)
