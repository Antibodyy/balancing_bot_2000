# make sure every file can run on any computers, avoid absolute paths and undownloaded packages

#pip install setuptools first if not installed
from setuptools import setup, find_packages
import os

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        "pyyaml",
        "numpy",
        "scipy",
        "matplotlib",
        "mujoco",
    ]

setup(
    name="balancing_bot_2000",
    version="0.1.0",
    description="MPC Self-Balancing Robot",
    packages=find_packages(include=[
        'robot_dynamics',
        'robot_dynamics.*',
        'mpc',
        'mpc.*',
        'simulation',
        'simulation.*',
        'state_estimation',
        'state_estimation.*',
        'control_pipeline',
        'control_pipeline.*',
        'config',
        'config.*',
    ]),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    package_data={
        '': ['*.xml', '*.yaml', '*.yml', '*.json'],
    },
    include_package_data=True,
)
