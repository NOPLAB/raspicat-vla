from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'raspicat_vla_remote'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'grpcio>=1.50'],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='nop@example.com',
    description='VLA remote gRPC server (model-agnostic).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_dummy_server = raspicat_vla_remote.server_main:main',
        ],
    },
)
