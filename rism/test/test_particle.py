#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_particle.py
created time : 2023/05/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import numpy as np
from rism.core import Particle
from rism.unit import *


class TestParticle:
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_attribute(self):
        particle = Particle(
            "O",
            sigma=Quantity(0.1, nanometer),
            epsilon=Quantity(0.5, kilocalorie_permol),
            q=Quantity(1, elementary_charge),
        )
        assert np.isclose(particle.sigma, 1)
        assert (
            particle.epsilon
            == Quantity(0.5, kilocalorie_permol).convert_to(default_energy_unit).value
        )
        assert np.isclose(particle.q, 1)

    def test_exceptions(self):
        pass

    def test_asdict(self):
        particle = Particle(
            "O",
            sigma=Quantity(0.1, nanometer),
            epsilon=Quantity(0.5, kilocalorie_permol),
            q=Quantity(1, elementary_charge),
        )
        particle_1 = Particle(**particle.asdict())
        assert particle_1.name == particle.name
        assert particle_1.sigma == particle.sigma
        assert particle_1.epsilon == particle.epsilon
        assert particle_1.q == particle.q
