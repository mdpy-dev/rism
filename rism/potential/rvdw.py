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
from rism.core import Particle
from rism.unit import *
from rism.environment import *


class RVDWPotential:
    def __init__(self, particle1: Particle, particle2: Particle) -> None:
        self._particle1 = particle1
        self._particle2 = particle2
        self._sigma = CUPY_FLOAT(0.5 * (self._particle1.sigma + self._particle2.sigma))
        self._epsilon = CUPY_FLOAT(
            np.sqrt(self._particle1.epsilon * self._particle2.epsilon)
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
    def particle1(self) -> Particle:
        return self._particle1

    @property
    def particle2(self) -> Particle:
        return self._particle2

    @property
    def sigma(self):
        return self._sigma

    @property
    def epsilon(self):
        return self._epsilon
