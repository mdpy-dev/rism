#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : error.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


class UnitDimensionMismatchedError(Exception):
    """This error occurs when:
    - The base dimension of two quantities is mismatched for a specific operation.

    Used in:
    - mdpy.unit.base_dimension
    """

    pass
