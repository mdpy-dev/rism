#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : ele.py
created time : 2023/04/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import math
import cupy as cp
import numpy as np
from cupyx.scipy.special import erf, erfc
from rism.core import Particle
from rism.unit import *
from rism.environment import *


class ElePotential:
    def __init__(
        self,
        particle1: Particle,
        particle2: Particle,
        r_cut=Quantity(10, angstrom),
        direct_sum_energy_tolerance=1e-5,
        alpha=None,
    ) -> None:
        self._particle1 = particle1
        self._particle2 = particle2
        self._r_cut = check_quantity_value(r_cut, default_length_unit)
        self._direct_sum_energy_tolerance = direct_sum_energy_tolerance
        if alpha is None:
            self._alpha = self._get_ewald_coefficient()
        else:
            self._alpha = alpha
        self._q1, self._q2 = self._particle1.q, self._particle2.q
        self._epsilon0 = EPSILON0.convert_to(
            default_charge_unit**2 / default_energy_unit / default_length_unit
        ).value
        self._factor = CUPY_FLOAT(self._q1 * self._q2 / (4 * np.pi * self._epsilon0))

    def _get_ewald_coefficient(self):
        """
        Using Newton iteration to solve the ewald coefficient:

        The Ewald coefficient is essentially a mathematical representation of 1/2s, where s represents the width of the Gaussian function used to smooth out charges on the grid. The value of s is carefully chosen such that, at the interaction `r_cut`, the interactions between two Gaussian-smoothed charges and two point charges are equivalent up to a precision of direct_sum_energy_tolerance. This ensures that the interactions between these four charge sites are small enough, indicating that truncation of the interactions will not result in a significant error.

        f(alpha) = erfc(alpha*r_cut) / r_cut - direct_sum_energy_tolerance
        f'(alpha) = - 2/sqrt(pi) * exp[-(alpha*r_cut)^2]

        alpha_new = alpha_old - f(alpha_old) / f'(alpha_old)
        """
        alpha = 0.1
        sqrt_pi = np.sqrt(np.pi)
        while True:
            f = (
                math.erfc(alpha * self._r_cut) / self._r_cut
                - self._direct_sum_energy_tolerance
            )
            df = -2 * np.exp(-((alpha * self._r_cut) ** 2)) / sqrt_pi
            d_alpha = f / df
            if np.abs(d_alpha / alpha) < 1e-5:
                break
            alpha -= d_alpha
        return CUPY_FLOAT(alpha)

    def evaluate(self, dist, threshold=Quantity(10.0, kilocalorie_permol)):
        if not isinstance(dist, cp.ndarray):
            dist = cp.array(dist, CUPY_FLOAT)
        threshold = check_quantity_value(threshold, default_energy_unit)
        ele = self._factor / (dist + 1e-5)
        if threshold > 0:
            ele[ele > threshold] = threshold
            ele[ele < -threshold] = -threshold
        return ele.astype(CUPY_FLOAT)

    def evaluate_sr(self, dist, threshold=Quantity(10.0, kilocalorie_permol)):
        ele = self.evaluate(dist, threshold=threshold)
        factor = erfc(dist * self._alpha)
        return (ele * factor).astype(CUPY_FLOAT)

    def evaluate_lr(self, dist, threshold=Quantity(10.0, kilocalorie_permol)):
        ele = self.evaluate(dist, threshold=threshold)
        factor = erf(dist * self._alpha)
        return (ele * factor).astype(CUPY_FLOAT)

    def evaluate_lr_k(self, k):
        res = 4 * cp.pi * self._factor / k**2
        res *= cp.exp(-((k / 2 / self._alpha) ** 2))
        return res.astype(CUPY_FLOAT)

    @property
    def particle1(self) -> Particle:
        return self._particle1

    @property
    def particle2(self) -> Particle:
        return self._particle2
