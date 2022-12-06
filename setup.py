import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="talk",
    py_modules=["talk"],
    version="1.0",
    description="Librarized code to test it more easily on VM",
    readme="README.md",
    python_requires=">=3.7",
    author="Aszeles Sceznyk",
    url="https://github.com/asceznyk/talk",
    license="None",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True
) 






