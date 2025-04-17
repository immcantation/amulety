#!/usr/bin/env python

from setuptools import find_packages, setup

version = "1.1"

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="amulety",
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
    author="Mamie Wang, Gisela Gabernet, Steven Kleinstein",
    author_email="mamie.wang@yale.edu, gisela.gabernet@yale.edu, steven.kleinstein@yale.edu",
    url="https://github.com/immcantation/amulety",
    license="MIT",
    entry_points={
        "console_scripts": ["amulety=amulety.amulety:main"],
    },
    python_requires=">=3.8, <4",
    install_requires=required,
    packages=find_packages(exclude=("docs")),
    include_package_data=True,
    zip_safe=False,
)
