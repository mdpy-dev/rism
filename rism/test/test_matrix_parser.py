#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_matrix_parser.py
created time : 2023/05/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import os
import pytest
import cupy as cp
from rism.core import FFTGrid, Matrix
from rism.io import MatrixParser
from rism.error import SuffixError


class TestMatrixParser:
    def setup_method(self):
        grid = FFTGrid(x=[-10, 10, 64], y=[-10, 10, 64])
        site_list = ["o", "h1", "h2"]
        self.matrix = Matrix(grid, site_list)

    def teardown_method(self):
        del self.matrix

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(SuffixError):
            MatrixParser("test.ma")

    def test_parser(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(cur_dir, "out/test_matrix.mat")
        parser = MatrixParser(file_path)
        matrix = parser.parse()
        assert self.matrix._num_sites == matrix._num_sites
        assert self.matrix._num_data == matrix._num_data
        # Test site
        for i in range(matrix.num_sites):
            assert self.matrix.site_list[i] == matrix.site_list[i]
        # Test grid
        for i in range(matrix.grid.shape[0]):
            assert self.matrix.grid.x[i, 0] == matrix.grid.x[i, 0]
            assert self.matrix.grid.y[i, 0] == matrix.grid.y[i, 0]
        # Test data
        for i in range(matrix.grid.shape[0]):
            assert self.matrix[0, 0][i, 0] == matrix[0, 0][i, 0]
            assert self.matrix[2, 0][i, 0] == matrix[2, 0][i, 0]


if __name__ == "__main__":
    test = TestMatrixParser()
    test.setup_method()
    test.test_parser()
    test.teardown_method()
