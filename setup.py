r"""Basic setup module"""
from setuptools import setup, find_packages


with open("NeedALight/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = [
    "numpy",
    "scipy",
    "matplotlib",
    "pytest",
    "StrawberryFields",
    "thewalrus",
    "custom_poling",
    "jupyter"
]

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]


setup(
    name="NeedALight",
    version=version,
    description="twin beam generation",
    url="https://github.com/polyquantique/NeedALight",
    author="Martin Houde",
    author_email="martin.houde2@gmail.com",
    license="Apache License 2.0",
    packages=find_packages(where="."),
    install_requires=requirements,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    classifiers=classifiers,
)