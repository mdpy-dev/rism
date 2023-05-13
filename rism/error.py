#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : error.py
created time : 2023/04/13
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


class ClosureInputError(Exception):
    """This error occurs when:
    - The input of closure function is not supported.

    Used in:
    - rism.closure
    """

    pass


class ArrayShapeError(Exception):
    """This error occurs when:
    - The Array has a unmatched shape

    Used in:
    - rism.core.dcf
    """

    pass


class SuffixError(Exception):
    """This error occurs when:
    - The suffix of file is unmatched with the requirement

    Used in:
    - rism.io
    """

    pass


class DuplicatedNameError(Exception):
    """This error occurs when:
    - A list of particles have same particle name

    Used in:
    - rism.core.solvent
    """

    pass


class SolventIncompleteError(Exception):
    """This error occurs when:
    - The bonding information of solvent is not complete

    Used in:
    - rism.core.solvent
    """

    pass
