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
from rism.potential import ElePotential
from rism.unit import *
from rism.error import *


class TestVDWPotential:
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_attributes(self):
        u = ElePotential(type1="k", type2="c")
        assert u.type1 == "k"
        assert u.type2 == "c"

    def test_exceptions(self):
        with pytest.raises(UnregisteredParticleError):
            ElePotential(type1="ab", type2="c")

    def test_evaluate(self):
        threshold = Quantity(1, kilocalorie_permol)
        ele = ElePotential(type1="k", type2="k")
        r = cp.arange(0, 10, 0.1)
        u = ele.evaluate(r, threshold=threshold)
        assert isinstance(u, cp.ndarray)
        assert u.dtype == CUPY_FLOAT
        assert np.isclose(
            u.max().get(), threshold.convert_to(default_energy_unit).value
        )


if __name__ == "__main__":

    ele = ElePotential(type1="k", type2="o")
    print(ele.q1, ele.q2)
    r = cp.arange(1, 20, 0.1)
    u = ele.evaluate(r, threshold=Quantity(100, kilocalorie_permol))
    convert = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    plt.plot(r.get(), u.get() / convert, ".-")
    plt.show()
