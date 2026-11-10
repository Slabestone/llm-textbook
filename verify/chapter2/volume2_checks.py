#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二卷 v1.0 教学图算术复核；不代表真实 GPU 基准。"""
from math import isclose

assert isclose(10/24, 0.4166666666666667)
assert isclose(10/12, 0.8333333333333334)
assert isclose((1**(-.3) + 1**(-.3)), 2)
assert isclose((.25**(-.3) + 4**(-.3)), (4**(-.3) + .25**(-.3)))
assert isclose(32**(.2/.5), 4)
assert isclose(32**(.3/.5), 8)
assert sum((1,8,23)) == 32
assert sum((1,5,10)) == 16
assert sum((1,8,7)) == 16
assert sum((2,2,4,4,4)) == 16
assert 7_000_000_000*16 == 112_000_000_000
assert (2+10)/2 == 6
assert (1*2+3*10)/(1+3) == 8
assert isclose(2*(8-1)/8*2/100, .035)
assert isclose(5/8, .625)
assert isclose(80/14/8, 5/7)
print("PASS: Packing / Scaling / FP bitfields / memory ledger / gradient / All-Reduce / strong scaling")
