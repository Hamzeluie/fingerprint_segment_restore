#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="mehdi hamzeluie",
    author_email='mehdihamze73@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="reconstruct deformed fingerprint",
    entry_points={
        'console_scripts': [
            'fingerprint_segment_restore=fingerprint_segment_restore.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fingerprint_segment_restore',
    name='fingerprint_segment_restore',
    packages=find_packages(include=['fingerprint_segment_restore', 'fingerprint_segment_restore.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Hamzeluie/fingerprint_segment_restore',
    version='0.1.0',
    zip_safe=False,
)
