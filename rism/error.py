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


class UnregisteredParticleError(Exception):
    """This error occurs when:
    - A particle type has not been registered in rism.FFDict is provided.

    Used in:
    - rism.potential
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
