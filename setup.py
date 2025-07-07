#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='synthetics',
    version=open("version.txt").read().rstrip(),
    description='Generation of synthetic faces',
    license='Custom',
    author='David Geissbühler, Laurent Colbois',
    author_email='david.geissbuhler@idiap.ch',
    maintainer='David Geissbühler',
    maintainer_email='david.geissbuhler@idiap.ch',
    long_description=open('README.md').read(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe = False,
    install_requires=['setuptools'],
    entry_points={"console_scripts" : ['synthetics = synthetics.main:cli']},
    classifiers = 
    [
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
