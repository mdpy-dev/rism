#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : solvent.py
created time : 2023/05/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

from rism.core import Solvent


def tip3p():
    solvent = Solvent()
    solvent.add_particle(name="o", particle_type="o")
    solvent.add_particle(name="h1", particle_type="h")
    solvent.add_particle(name="h2", particle_type="h")
    solvent.add_bond("o", "h1", 0.9572)
    solvent.add_bond("o", "h2", 0.9572)
    solvent.add_bond("h1", "h2", 1.5139)
    return solvent
