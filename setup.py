from setuptools import find_packages, setup
exec(open('rapa/_version.py').read())




setup(
    name='rapa',
    packages=find_packages(),
    version=__version__,
    description='Robust Automtated Parsimony Analysis',
    long_description=open('README.md').read(),
    author='Life Epigenetics',
    author_email='info@FOXOBioScience.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
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
    install_requires=[
        'numpy',
        'pandas >=1.3.0',
        'scikit-learn',
        'matplotlib',
        'datarobot',
        'tqdm',
        'pickle',
        'seaborn'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    url='https://github.com/FoxoTech/rapa',
)