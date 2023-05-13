#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_solvent.py
created time : 2023/04/20
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import os
import pytest
from rism.core import Solvent
from rism.error import DuplicatedNameError, SolventIncompleteError
from rism.element import *
from rism.unit import *


class TestSolvent:
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_attribute(self):
        solvent = Solvent([])
        assert solvent.num_particles == 0
        assert solvent.particle_list == []
        assert solvent.num_bonds == 0
        assert solvent.bond_list == []

    def test_exceptions(self):
        with pytest.raises(DuplicatedNameError):
            solvent = Solvent([hydrogen(), hydrogen()])

        with pytest.raises(SolventIncompleteError):
            solvent = Solvent([oxygen("o"), hydrogen("h1"), hydrogen("h2")])
            solvent.add_bond("o", "h1", Quantity(0.1, nanometer))
            solvent.check_completeness()

    def test_parse_index(self):
        solvent = Solvent([oxygen("o"), hydrogen("h1"), hydrogen("h2")])
        assert solvent._parse_index("o") == 0
        assert solvent._parse_index(solvent.particle_list[1]) == 1
        assert solvent._parse_index(2) == 2

        with pytest.raises(ValueError):
            solvent._parse_index("a")

        with pytest.raises(ValueError):
            solvent._parse_index(oxygen("o1"))

        with pytest.raises(IndexError):
            solvent._parse_index(4)

        with pytest.raises(KeyError):
            solvent._parse_index(None)

    def test_add_bond(self):
        solvent = Solvent([oxygen("o"), hydrogen("h1"), hydrogen("h2")])
        solvent.add_bond("o", "h1", Quantity(0.1, nanometer))

        bond = solvent.get_bond_length("o", "h1")
        assert bond == 1

        bond = solvent.get_bond_length("h1", "o")
        assert bond == 1

        bond = solvent.get_bond_length(1, 0)
        assert bond == 1


if __name__ == "__main__":
    solvent = Solvent([oxygen("o"), hydrogen("h1"), hydrogen("h2")])
    solvent.add_bond("o", "h1", 0.9572)
    solvent.add_bond("o", "h2", 0.9572)
    solvent.add_bond("h1", "h2", 1.5139)
    print(solvent.asdict())
