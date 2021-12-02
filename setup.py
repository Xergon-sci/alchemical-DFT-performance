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
    project_urls={
        'Bug Tracker': 'https://github.com/Xergon-sci/alchemical-DFT-performance/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'adftp = adft_performance.cli:cli',
        ],
    },
)
