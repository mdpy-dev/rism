#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : closure.py
created time : 2023/04/14
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
from rism.environment import CUPY_FLOAT


def percus_yevick(exp_u, gamma):
    c = (exp_u - CUPY_FLOAT(1)) * (gamma + CUPY_FLOAT(1))
    return c.astype(CUPY_FLOAT)


def hnc(exp_u, gamma):
    c = exp_u * cp.exp(gamma) - gamma - CUPY_FLOAT(1)
    return c.astype(CUPY_FLOAT)
