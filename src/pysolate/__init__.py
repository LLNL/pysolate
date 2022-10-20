# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# LLNL-CODE-841231
# authors:
#        Ana Aguiar (aguiarmoya1@llnl.gov)
#        Andrea Chiang (andrea@llnl.gov)
"""
pySolate - Seismic Data Processing Tool in Python
=====================================================

pysolate is a tool for processing seismic data using continuous wavelet transform (CWT).
The tool is based on the Matlab software Block Choice Seismic Analysis (BCseis, version 1.1)
by Charles A. Langston and S. Mostafa Mousavi.

:copyright:
    Copyright (c) 2020, Lawrence Livermore National Security, LLC
:license:
    LGPL-3.0 License
"""

# Generic release markers:
# X.Y
# X.Y.Z # bug fix and minor updates

# dev branch marker is "X.Y.devN" where N is an integer.
# X.Y.dev0 is the canonical version of X.Y.dev

__version__ = "1.0.0"

from .block import Parameter, Block, read
