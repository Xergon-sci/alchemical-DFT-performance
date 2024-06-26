""" Used for setup.
"""

from setuptools import setup
from setuptools import find_packages

setup(
    name='adft-performance',
    version='0.0.1',
    author='Michiel Jacobs',
    author_email='michiel.jacobs@vub.be',
    description='The power version of alchemical DFT for installation on HPC infrastructure.',
    url='https://github.com/Xergon-sci/alchemical-DFT-performance',
    packages=find_packages(),
    python_requires='>=3.6',
)