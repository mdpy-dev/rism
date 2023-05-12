#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_matrix_writer.py
created time : 2023/05/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import os
import pytest
import cupy as cp
from rism.core import FFTGrid, Matrix
from rism.io import MatrixWriter
from rism.error import SuffixError


class TestMatrixWriter:
    def setup_method(self):
        grid = FFTGrid(x=[-10, 10, 64], y=[-10, 10, 64])
        site_list = ["o", "h1", "h2"]
        self.matrix = Matrix(grid, site_list)

    def teardown_method(self):
        del self.matrix

    def test_attributes(self):
        assert self.matrix._num_sites == 3
        assert self.matrix._num_data == 6

    def test_exceptions(self):
        with pytest.raises(SuffixError):
            MatrixWriter("test.ma")

    def test_write(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(cur_dir, "out/test_matrix.mat")
        writer = MatrixWriter(file_path)
        writer.write(self.matrix)
