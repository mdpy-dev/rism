#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_fun_matrix.py
created time : 2023/05/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import pytest
import cupy as cp
from rism.core import FFTGrid, FunMatrix
from rism.error import ArrayShapeError


class TestFunMatrix:
    def setup_method(self):
        grid = FFTGrid(x=[-10, 10, 64], y=[-10, 10, 64])
        site_list = ["o", "h1", "h2"]
        self.matrix = FunMatrix(grid, site_list)

    def teardown_method(self):
        del self.matrix

    def test_attributes(self):
        assert self.matrix._num_sites == 3
        assert self.matrix._num_data == 6

    def test_exceptions(self):
        with pytest.raises(ArrayShapeError):
            self.matrix[0, 0] = cp.ones([12])

        with pytest.raises(ArrayShapeError):
            self.matrix[0, 0] = cp.ones([12])

    def test_parse_site(self):
        assert self.matrix._parse_site("h1") == 1
        assert self.matrix._parse_site(1) == 1

        with pytest.raises(IndexError):
            self.matrix._parse_site(4)

        with pytest.raises(KeyError):
            self.matrix._parse_site(None)

        with pytest.raises(ValueError):
            self.matrix._parse_site("A")

    def test_parse_site_pair(self):
        assert self.matrix._parse_site_pair(0, 0) == 0
        assert self.matrix._parse_site_pair(0, 1) == 1
        assert self.matrix._parse_site_pair(0, 2) == 2
        assert self.matrix._parse_site_pair(1, 1) == 3
        assert self.matrix._parse_site_pair(1, 2) == 4
        assert self.matrix._parse_site_pair(2, 2) == 5

        assert self.matrix._parse_site_pair(1, 2) == self.matrix._parse_site_pair(2, 1)
        assert self.matrix._parse_site_pair(2, 0) == self.matrix._parse_site_pair(0, 2)

    def test_set_get_item(self):
        a = cp.ones(self.matrix.grid.shape)
        self.matrix[0, 0] = a
        assert self.matrix[0, 0][0, 0] == a[0, 0]

        a[0, 0] = 2
        assert self.matrix[0, 0][0, 0] == 1
        assert self.matrix[0, 0][0, 0] != a[0, 0]


if __name__ == "__main__":
    test = TestFunMatrix()
    test.setup_method()
    test.test_parse_site_pair()
    test.matrix[0, 1] = cp.ones([64, 64])
    print(test.matrix[0, 1])
    test.teardown_method()
