from setuptools import setup
import os
from glob import glob

package_name = 'raspicat_async_vla_remote'

setup(
    name=package_name,
    version='0.1.0',
    packages=['asyncvla_remote'],
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
    description='AsyncVLA remote gRPC server.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'asyncvla_dummy_server = asyncvla_remote.server_main:main',
        ],
    },
)
