#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : utils.py
created time : 2023/04/13
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-present, Zhenyu Wei and Southeast University
"""

from rism.unit.unit import Unit
from rism.unit.quantity import Quantity


def check_quantity(val, target_unit: Unit):
    if isinstance(val, Quantity):
        return val.convert_to(target_unit)
    elif isinstance(val, type(None)):
        return None
    else:
        return Quantity(val, target_unit)


def check_quantity_value(val, target_unit: Unit):
    if isinstance(val, Quantity):
        return val.convert_to(target_unit).value
    elif isinstance(val, type(None)):
        return None
    else:
        return Quantity(val, target_unit).value
