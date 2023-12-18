from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib/python3.10/site-packages', package_name, 'signs_to_detect'), glob(os.path.join('signs_to_detect', '*.png'))),
        (os.path.join('lib/python3.10/site-packages', package_name, 'generative_signs'), glob(os.path.join('generative_signs', '*.png'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alexey',
    maintainer_email='alexey@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'signs_detection = detection.signs_detection:main',
            'generated_signs_detection = detection.generated_signs_detection:main',
            'traffic_light = detection.traffic_light:main',
        ],
    },
)
