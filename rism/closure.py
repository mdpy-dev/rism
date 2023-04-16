#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : closure.py
created time : 2023/04/14
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
import torch as tc
from rism.environment import CUPY_FLOAT
from rism.error import ClosureInputError

ERROR_TEMPLATE = "Only cp.ndarray and tc.Tensor supported, while (%s, %s) is provided"


def percus_yevick(exp_u, gamma):
    if isinstance(exp_u, cp.ndarray):
        c = (exp_u - CUPY_FLOAT(1)) * (gamma + CUPY_FLOAT(1))
        return c.astype(CUPY_FLOAT)
    elif isinstance(exp_u, tc.Tensor):
        c = (exp_u - 1) * (gamma + 1)
        return c
    else:
        raise ClosureInputError(ERROR_TEMPLATE % (type(exp_u), type(gamma)))


def hnc(exp_u, gamma):
    if isinstance(exp_u, cp.ndarray):
        c = exp_u * cp.exp(gamma) - gamma - CUPY_FLOAT(1)
        return c.astype(CUPY_FLOAT)
    elif isinstance(exp_u, tc.Tensor):
        c = exp_u * tc.exp(gamma) - gamma - 1
        return c
    else:
        raise ClosureInputError(ERROR_TEMPLATE % (type(exp_u), type(gamma)))


def kovalenko_hirata(exp_u, gamma):
    if isinstance(exp_u, cp.ndarray):
        u = cp.log(exp_u)
        chi = u + gamma
        area = chi <= 0
        chi[area] = cp.exp(chi[area]) - CUPY_FLOAT(1)
        c = chi - gamma
        return c.astype(CUPY_FLOAT)
    elif isinstance(exp_u, tc.Tensor):
        u = tc.log(exp_u)
        chi = u + gamma
        area = chi <= 0
        chi[area] = tc.exp(chi[area]) - 1
        c = chi - gamma
        return c
    else:
        raise ClosureInputError(ERROR_TEMPLATE % (type(exp_u), type(gamma)))
