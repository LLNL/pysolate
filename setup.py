# SPDX-License-Identifier: (BSD-3)
# LLNL-CODE-805542
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:
    Andrea Chiang (andrea@llnl.gov), 2022
    Ana Aguiar (aguiarmoya1@llnl.gov), 2022
"""
import codecs
import os
import re
from setuptools import find_packages, setup


NAME = "pysolate"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "pysolate", "__init__.py")
KEYWORDS = ["seismology","inversion","source"]
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: LGPL License",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Physics",
]
INSTALL_REQUIRES =[
    "obspy >= 1.2.0",
]


HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


VERSION = "1.0.0"


if __name__ == "__main__":
    setup(
        name=NAME,
        description="seismic data processing tool using the continuous wavelet transform",
        license="LGPL License",
        url="https://github.com/LLNL/pysolate",
        version=VERSION,
        author="Andrea Chiang",
        author_email="andrea@llnl.gov",
        keywords=KEYWORDS,
        long_description_content_type="text/x-rst",
        packages=PACKAGES,
        package_dir={"": "src"},
        python_requires=">=3.8.*",
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        options={"bdist_wheel": {"universal": "1"}},
    )
