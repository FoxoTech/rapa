from setuptools import find_packages, setup

exec(open('rapa/version.py').read())

with open('requirements.txt') as f: # get requirements from requirements.txt
    req = [x.strip('\n') for x in f.readlines()]

with open('tests/test_requirements.txt') as f: # get test requirements from test_requirements.txt
    test_req = [x.strip('\n') for x in f.readlines()]

setup(
    name='rapa',
    version=__version__,
    description='Robust Automtated Parsimony Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='FOXO Technologies',
    author_email='info@foxotechnologies.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
      ],
    keywords='feature reduction features datarobot robust automated parsimony analysis',
    install_requires=req,
    setup_requires=['pytest-runner'],
    tests_require=test_req,
    test_suite='tests',
    url='https://github.com/FoxoTech/rapa',
    packages=find_packages(),
    python_requires=">=3.8",
)