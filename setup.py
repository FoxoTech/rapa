from setuptools import find_packages, setup
import re

def get_version():
    return(re.findall(r"__version__.+\"(.+)\"",open("rapa/_version.py", 'r').read())[0])

rapa_version = get_version()
print(rapa_version)

setup(
    name='rapa',
    packages=find_packages(),
    version=rapa_version,
    description='Robust Automtated Parsimony Analysis',
    author='RO, JS',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    test_require=['pytest'],
    test_suite='tests',
)