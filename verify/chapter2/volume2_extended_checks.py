"""第二卷 II-03：原创教学图的确定性复算（不是对真实论文的重新拟合）。"""
from math import isclose, log10

alpha = 0.30
beta = 0.30

def excess_loss(n, d):
    assert n > 0 and d > 0
    return n**(-alpha) + d**(-beta)

for n, d, target in [(0.25, 4, 2.175470521896845),
                     (1, 1, 2.0),
                     (4, 0.25, 2.175470521896845)]:
    assert isclose(n*d, 1.0, abs_tol=1e-12)
    assert isclose(excess_loss(n, d), target, abs_tol=1e-12)

for j in range(-200, 201):
    n = 10**(j / 100)
    assert excess_loss(n, 1/n) >= 2 - 1e-12

for x in (0, 0.5, 1, 2, 4):
    n = 10**x
    assert isclose(log10(n**(-alpha)), -alpha*x, abs_tol=1e-12)

a, b = 0.3, 0.2
assert isclose(32**(b/(a+b)), 4.0, rel_tol=1e-12)
assert isclose(32**(a/(a+b)), 8.0, rel_tol=1e-12)
print("PASS: 3 normalized budget points; loss minimum; log-log slope; budget exponents")

