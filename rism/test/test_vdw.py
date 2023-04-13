#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_vdw.py
created time : 2023/03/23
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import cupy as cp
import matplotlib.pyplot as plt
from rism.potential import VDWPotential
from rism.unit import *

if __name__ == "__main__":

    vdw = VDWPotential(type1="c", type2="k")
    r = cp.arange(0, 10, 0.1)
    u = vdw.evaluate(r, threshold=Quantity(1, kilocalorie_permol))
    plt.plot(r.get(), u.get(), ".-")
    plt.show()
