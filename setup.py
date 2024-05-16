#!/usr/bin/env python

from setuptools import find_packages, setup

version = "0.1.0"

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="bcrembed",
    version=version,
    description="Python package to create embeddings of BCR amino acid sequences.",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=[
        "immcantation",
        "immunoinformatics",
        "bioinformatics",
        "embedding",
        "antibody",
        "BCR",
        "Machine Learning",
        "biology",
        "NGS",
        "next generation sequencing",
    ],
    author="Mamie Wang",
    author_email="mamie.wang@yale.edu",
    url="https://github.com/immcantation/bcrembed",
    license="MIT",
    entry_points={
        "console_scripts": ["bcrembed=bcrembed.bcrembed:main"],
    },
    python_requires=">=3.8, <4",
    install_requires=required,
    packages=find_packages(exclude=("docs")),
    include_package_data=True,
    zip_safe=False,
)
