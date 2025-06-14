#! /usr/bin/env python
import sympy as sp


def bashforth(order: int) -> list[sp.Rational]:
    s = order
    M = [[(-j) ** k for j in range(s)] for k in range(s)]
    M = sp.Matrix(M)
    b = sp.Matrix([sp.Rational(1, k + 1) for k in range(s)])
    a = M.solve(b)
    return [sp.Rational(a_i) for a_i in a]


print(tuple(tuple(bashforth(o)) for o in range(1, 6)))
