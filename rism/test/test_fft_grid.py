#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_fft_grid.py
created time : 2023/04/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import pytest
import cupy as cp
from rism.core import FFTGrid


class TestFFTGrid:
    def setup(self):
        self.grid = FFTGrid(x=[-2, 2, 128], y=[-2, 2, 128], z=[-2, 2, 128])

    def teardown(self):
        del self.grid

    def test_attribute(self):
        assert hasattr(self.grid, "x")
        assert hasattr(self.grid, "y")
        assert hasattr(self.grid, "z")
        assert hasattr(self.grid, "dx")
        assert hasattr(self.grid, "dy")
        assert hasattr(self.grid, "dz")
        assert hasattr(self.grid, "lx")
        assert hasattr(self.grid, "ly")
        assert hasattr(self.grid, "lz")

        assert cp.isclose(self.grid.dx, (self.grid.x[1, 0, 0] - self.grid.x[0, 0, 0]))

    def test_exception(self):
        pass


if __name__ == "__main__":
    test = TestFFTGrid()
    test.setup()
    test.test_attribute()
    test.teardown()
