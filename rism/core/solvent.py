#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : solvent.py
created time : 2023/04/20
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import json
from rism.potential import check_particle_type, FF_DICT
from rism.unit import *


class Solvent:
    def __init__(self, solvent_file_path: str = None) -> None:
        if solvent_file_path is None:
            self._num_particles = 0
            self._num_bonds = 0
            self._topology = {"particle": {}, "bond": {}}
        else:
            with open(solvent_file_path, "r") as f:
                self._topology = json.load(f)
            self._num_particles = len(self.particle_list)
            self._num_bonds = len(self.bond_list)

    def _exists_particle(self, name):
        if name in self.particle_list:
            return True
        return False

    def _check_particle(self, name):
        if not self._exists_particle(name):
            raise KeyError("Particle %s does not exists in the topology" % name)

    def add_particle(self, name, particle_type, particle_params: dict = None):
        particle_type = check_particle_type(particle_type)
        if not self._exists_particle(name) or not particle_params is None:
            cur_particle = {"type": particle_type}
            if particle_params is None:
                particle_params = FF_DICT[particle_type]
            cur_particle["sigma"] = float(
                check_quantity_value(particle_params["sigma"], default_length_unit)
            )
            cur_particle["epsilon"] = float(
                check_quantity_value(particle_params["epsilon"], default_energy_unit)
            )
            cur_particle["q"] = float(
                check_quantity_value(particle_params["q"], default_charge_unit)
            )
            self._topology["particle"][name] = cur_particle
        self._num_particles = len(self.particle_list)

    def get_particle(self, name):
        self._check_particle(name)
        return self._topology["particle"][name]

    def _exists_bond(self, name1, name2):
        cur_bond_name = self._topology["bond"].keys()
        bond_names = ["%s-%s" % (name1, name2)]
        bond_names += ["%s-%s" % (name2, name1)]
        for bond_name in bond_names:
            if bond_name in cur_bond_name:
                return True, bond_name
        return False, bond_name

    def add_bond(self, name1, name2, bond_length):
        self._check_particle(name1)
        self._check_particle(name2)
        bond_name = "%s-%s" % (name1, name2)
        if not self._exists_bond(name1, name2)[0]:
            self._topology["bond"][bond_name] = float(
                check_quantity_value(bond_length, default_length_unit)
            )
        self._num_bonds = len(self.bond_list)

    def get_bond(self, name1, name2):
        is_exist, bond_name = self._exists_bond(name1, name2)
        if is_exist:
            return self._topology["bond"][bond_name]
        else:
            raise KeyError(
                "Bond %s-%s does not exists in the topology" % (name1, name2)
            )

    def save(self, solvent_file_path: str):
        with open(solvent_file_path, "w") as f:
            data = json.dumps(self._topology, sort_keys=True, indent=4)
            data = data.encode("utf-8").decode("unicode_escape")
            print(data, file=f)

    @property
    def topology(self) -> dict:
        return self._topology

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def particle_list(self) -> list:
        return list(self._topology["particle"].keys())

    @property
    def num_bonds(self) -> int:
        return self._num_bonds

    @property
    def bond_list(self) -> list:
        return list(self._topology["bond"].keys())
