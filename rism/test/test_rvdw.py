#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_rvdw.py
created time : 2023/04/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import pytest
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from rism.environment import CUPY_FLOAT
from rism.potential import RVDWPotential
from rism.unit import *
from rism.error import *


class TestRVDWPotential:
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_attributes(self):
        u = RVDWPotential(type1="k", type2="c")
        assert u.type1 == "k"
        assert u.type2 == "c"

    def test_exceptions(self):
        with pytest.raises(UnregisteredParticleError):
            RVDWPotential(type1="ab", type2="c")

    def test_evaluate(self):
        threshold = Quantity(1, kilocalorie_permol)
        vdw = RVDWPotential(type1="c", type2="k")
        r = cp.arange(0, 10, 0.1)
        u = vdw.evaluate(r, threshold=threshold)
        assert isinstance(u, cp.ndarray)
        assert u.dtype == CUPY_FLOAT
        assert np.isclose(
            u.max().get(), threshold.convert_to(default_energy_unit).value
        )


if __name__ == "__main__":

    vdw = RVDWPotential(type1="c", type2="k")

    r = cp.arange(0, 5, 0.1)
    u = vdw.evaluate(r, threshold=Quantity(1, kilocalorie_permol))
    plt.plot(r.get(), u.get(), ".-")
    plt.show()
