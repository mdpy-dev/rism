#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : particle.py
created time : 2023/05/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


from rism.unit import *


class Particle:
    def __init__(
        self, name: str, sigma: Quantity, epsilon: Quantity, q: Quantity
    ) -> None:
        self._name = name
        self._sigma = check_quantity_value(sigma, default_length_unit)
        self._epsilon = check_quantity_value(epsilon, default_energy_unit)
        self._q = check_quantity_value(q, default_charge_unit)

    def asdict(self):
        return {
            "name": self._name,
            "sigma": self._sigma,
            "epsilon": self._epsilon,
            "q": self._q,
        }

    @property
    def name(self):
        return self._name

    @property
    def sigma(self):
        return self._sigma

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def q(self):
        return self._q
