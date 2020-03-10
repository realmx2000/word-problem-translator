from sympy import Symbol, solve, Eq, diff, lambdify
from sympy.abc import _clash1
from sympy.parsing.sympy_parser import parse_expr

from collections import defaultdict

class Solver:
    @staticmethod
    def substitute(equation: str, const_dict: dict):
        for const_name, const_val in const_dict.items():
            equation = equation.replace(const_name, str(const_val))
        return equation

    @staticmethod
    def convert(equation: str):
        eq_idx = equation.find('=')
        lhs = equation[:eq_idx]
        rhs = equation[eq_idx + 1:]
        lhs_sym = parse_expr(lhs, local_dict=_clash1)
        rhs_sym = parse_expr(rhs, local_dict=_clash1)
        return lhs_sym - rhs_sym

    @staticmethod
    def solve(equations, var_dict, const_dict):
        equations_sym = []
        for equation in equations:
            equation_sym = Solver.convert(equation)
            equations_sym.append(equation_sym)
        var_symbols = [Symbol(var) for var in var_dict.keys()]
        fn_dict = solve(equations_sym, var_symbols)
        solution_dict = {}
        for i, var in enumerate(var_dict.keys()):
            solution_dict[var] = lambdify(list(const_dict.keys()), fn_dict[var_symbols[i]])(*list(const_dict.values()))

        derivative_dict = Solver.get_derivatives(fn_dict, const_dict)
        print(fn_dict)
        return solution_dict, derivative_dict

    @staticmethod
    def get_derivatives(fn_dict, const_dict):
        derivative_dict = defaultdict(dict)
        for var, equation in fn_dict.items():
            for const in const_dict.keys():
                derivative_dict[var][const] = lambdify(list(const_dict.keys()), diff(equation, Symbol(const)))(*list(const_dict.values()))
        return derivative_dict

if __name__ == "__main__":
    #equation = input("Equation: ")
    import cProfile
    import pstats
    equations = ['a*x+b*y=c', 'd*x+e*y=f']
    pr = cProfile.Profile()
    pr.enable()
    print(Solver.solve(equations, {'x': None, 'y': None}, {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6}))
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative').print_stats(50)

