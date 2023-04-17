#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : ele.py
created time : 2023/04/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import cupy as cp
import numpy as np
from rism.potential import FF_DICT, check_particle_type
from rism.unit import *
from rism.environment import *


class ElePotential:
    def __init__(self, type1: str, type2: str) -> None:
        self._type1 = check_particle_type(type1)
        self._type2 = check_particle_type(type2)
        self._q1, self._q2 = self._get_ele_parameter()
        self._epsilon0 = EPSILON0.convert_to(
            default_charge_unit**2 / default_energy_unit / default_length_unit
        ).value
        self._factor = CUPY_FLOAT(self._q1 * self._q2 / (4 * np.pi * self._epsilon0))

    def _get_ele_parameter(self):
        q1 = check_quantity_value(FF_DICT[self._type1]["val"], default_charge_unit)
        q2 = check_quantity_value(FF_DICT[self._type2]["val"], default_charge_unit)
        return (CUPY_FLOAT(q1), CUPY_FLOAT(q2))

    def evaluate(self, dist, threshold=Quantity(10.5, kilocalorie_permol)):
        if not isinstance(dist, cp.ndarray):
            dist = cp.array(dist, CUPY_FLOAT)
        threshold = check_quantity_value(threshold, default_energy_unit)
        ele = self._factor / (dist + 1e-10)
        if threshold > 0:
            ele[ele > threshold] = threshold
            ele[ele < -threshold] = -threshold
        return ele.astype(CUPY_FLOAT)

    @property
    def type1(self):
        return self._type1

    @property
    def type2(self):
        return self._type2

    @property
    def q1(self):
        return self._q1

    @property
    def q2(self):
        return self._q2
