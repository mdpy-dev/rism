#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : rvdw.py
created time : 2023/04/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import cupy as cp
import numpy as np
from rism.potential import FF_DICT, check_particle_type
from rism.unit import *
from rism.environment import *


class RVDWPotential:
    def __init__(self, type1: str, type2: str) -> None:
        self._type1 = check_particle_type(type1)
        self._type2 = check_particle_type(type2)
        self._sigma, self._epsilon = self._get_vdw_parameter()

    def _get_vdw_parameter(self):
        sigma1 = check_quantity_value(
            FF_DICT[self._type1]["sigma"], default_length_unit
        )
        epsilon1 = check_quantity_value(
            FF_DICT[self._type1]["epsilon"], default_energy_unit
        )
        sigma2 = check_quantity_value(
            FF_DICT[self._type2]["sigma"], default_length_unit
        )
        epsilon2 = check_quantity_value(
            FF_DICT[self._type2]["epsilon"], default_energy_unit
        )
        return (
            CUPY_FLOAT(0.5 * (sigma1 + sigma2)),
            CUPY_FLOAT(np.sqrt(epsilon1 * epsilon2)),
        )

    def evaluate(self, dist, threshold=Quantity(10.5, kilocalorie_permol)):
        if not isinstance(dist, cp.ndarray):
            dist = cp.array(dist, CUPY_FLOAT)
        threshold = check_quantity_value(threshold, default_energy_unit)
        scaled_distance = (self._sigma / (dist + 1e-5)) ** 6
        vdw = 4 * self._epsilon * (scaled_distance**2 - scaled_distance)
        vdw += self._epsilon
        if threshold > 0:
            vdw[vdw > threshold] = threshold
        r_min = self._sigma * 2 ** (1 / 6)
        vdw[dist > r_min] = 0
        return vdw.astype(CUPY_FLOAT)

    @property
    def type1(self):
        return self._type1

    @property
    def type2(self):
        return self._type2

    @property
    def sigma(self):
        return self._sigma

    @property
    def epsilon(self):
        return self._epsilon
