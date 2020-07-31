from setuptools import setup

setup(
    name='therepy',
    version='0.1',
    description='Module for building optimizers using data miners',
    author='Tim Menzies',
    author_email='timm@ieee.org',
    url='http://menzies.us/therepy',
    packages=['therepy'],
    long_description="""\
        Optimizer, written as a data miner. Break the 
        data up into regions of 'bad' and 'better'. 
        Find ways to jump from 'bad' to 'better'. 
        Nearly all this processing takes loglinear time.
      """,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Environment :: MacOS X",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
        "Natural Language :: English",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords='cluster contrast multi-objective-optimization decision-tree',
    license='MIT',
)
