#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_vdw.py
created time : 2023/04/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import pytest
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from rism.environment import CUPY_FLOAT
from rism.potential import VDWPotential
from rism.element import *
from rism.unit import *


class TestVDWPotential:
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_attributes(self):
        u = VDWPotential(potassium(), carbon())
        assert u.particle1.name == "k"
        assert u.particle2.name == "c"

    def test_exceptions(self):
        pass

    def test_evaluate(self):
        threshold = Quantity(1, kilocalorie_permol)
        vdw = VDWPotential(carbon(), potassium())
        r = cp.arange(0, 10, 0.1)
        u = vdw.evaluate(r, threshold=threshold)
        assert isinstance(u, cp.ndarray)
        assert u.dtype == CUPY_FLOAT
        assert np.isclose(
            u.max().get(), threshold.convert_to(default_energy_unit).value
        )


if __name__ == "__main__":

    vdw = VDWPotential(carbon(), potassium())
    r = cp.arange(0, 10, 0.1)
    u = vdw.evaluate(r, threshold=Quantity(1, kilocalorie_permol))
    plt.plot(r.get(), u.get(), ".-")
    plt.show()
