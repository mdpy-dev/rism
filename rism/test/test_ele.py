#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_ele.py
created time : 2023/04/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import pytest
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from rism.environment import CUPY_FLOAT
from rism.potential import ElePotential, VDWPotential
from rism.element import *
from rism.unit import *


class TestElePotential:
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_attributes(self):
        u = ElePotential(potassium(), carbon())
        assert u.particle1.name == "k"
        assert u.particle2.name == "c"

    def test_exceptions(self):
        pass

    def test_evaluate(self):
        threshold = Quantity(1, kilocalorie_permol)
        ele = ElePotential(potassium(), potassium())
        r = cp.arange(0, 10, 0.1)
        u = ele.evaluate(r, threshold=threshold)
        assert isinstance(u, cp.ndarray)
        assert u.dtype == CUPY_FLOAT
        assert np.isclose(
            u.max().get(), threshold.convert_to(default_energy_unit).value
        )


if __name__ == "__main__":

    o = oxygen()
    ele = ElePotential(o, o, alpha=1.0)
    vdw = VDWPotential(o, o)
    r = cp.arange(1, 20, 0.1)
    u_long = ele.evaluate_lr(r, threshold=Quantity(-1, kilocalorie_permol))
    u_short = ele.evaluate_sr(r, threshold=Quantity(-1, kilocalorie_permol))
    u = ele.evaluate(r, threshold=Quantity(-1, kilocalorie_permol))
    u_vdw = vdw.evaluate(r, threshold=Quantity(-1, kilocalorie_permol))
    convert = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    plt.plot(r.get(), u.get() / convert, ".-", label="origin")
    plt.plot(r.get(), u_long.get() / convert, ".-", label="long range", alpha=0.5)
    plt.plot(r.get(), u_short.get() / convert, ".-", label="short range", alpha=0.5)
    plt.plot(
        r.get(),
        (u_short + u_vdw).get() / convert,
        ".-",
        label="short + vdw range",
        alpha=0.5,
    )

    plt.ylim(-1000, 1000)
    plt.legend()
    plt.show()
