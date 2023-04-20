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
from rism.core import Solvent, solvent
from rism.potential import FF_DICT
from rism.unit import *


class TestSolvent:
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_attribute(self):
        solvent = Solvent()
        assert solvent.num_particles == 0
        assert solvent.particle_list == []
        assert solvent.num_bonds == 0
        assert solvent.bond_list == []

    def test_exceptions(self):
        solvent = Solvent()

        with pytest.raises(KeyError):
            solvent.get_particle("a")

        with pytest.raises(KeyError):
            solvent.add_bond("a", "b", 1.0)

        with pytest.raises(KeyError):
            solvent.get_bond("a", "b")

    def test_add_particles(self):
        solvent = Solvent()

        solvent.add_particle(name="O", particle_type="o")
        assert solvent.num_particles == 1
        solvent.add_particle(name="O", particle_type="o")
        assert solvent.num_particles == 1

        particle = solvent.get_particle("O")
        params = FF_DICT["o"]
        assert particle["type"] == "o"
        assert particle["sigma"] == check_quantity_value(
            params["sigma"], default_length_unit
        )
        assert particle["epsilon"] == check_quantity_value(
            params["epsilon"], default_energy_unit
        )
        assert particle["q"] == check_quantity_value(params["q"], default_charge_unit)

        solvent.add_particle(
            name="O",
            particle_type="o",
            particle_params={"sigma": 1, "epsilon": 0, "q": 1},
        )
        particle = solvent.get_particle("O")
        assert solvent.num_particles == 1
        assert particle["sigma"] == 1
        assert particle["epsilon"] == 0
        assert particle["q"] == 1

    def test_add_bond(self):
        solvent = Solvent()
        solvent.add_particle(name="O", particle_type="o")
        solvent.add_particle(name="H1", particle_type="h")
        solvent.add_bond("O", "H1", Quantity(0.1, nanometer))

        bond = solvent.get_bond("O", "H1")
        assert bond == 1

        bond = solvent.get_bond("H1", "O")
        assert bond == 1

    def test_io(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        solvent_file_path = os.path.join(cur_dir, "out/test_solvent.json")

        solvent = Solvent()
        solvent.add_particle(name="O", particle_type="o")
        solvent.add_particle(name="H1", particle_type="h")
        solvent.add_bond("O", "H1", Quantity(0.1, nanometer))

        solvent.save(solvent_file_path)
        solvent_load = Solvent(solvent_file_path)

        assert solvent.num_particles == solvent_load.num_particles
        assert solvent.num_bonds == solvent_load.num_bonds
        for particle in ["O", "H1"]:
            particle = solvent.get_particle("O")
            particle_new = solvent_load.get_particle("O")
            assert particle["type"] == particle_new["type"]
            assert particle["epsilon"] == particle_new["epsilon"]
            assert particle["sigma"] == particle_new["sigma"]
            assert particle["q"] == particle_new["q"]
        bond = solvent.get_bond("H1", "O")
        bond_load = solvent.get_bond("O", "H1")
        assert bond == bond_load


if __name__ == "__main__":
    test = TestSolvent()
    test.test_add_bond()
