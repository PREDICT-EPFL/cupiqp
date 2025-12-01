import unittest
import numpy as np

import sys
sys.path.append('./')
sys.path.append('../')

from piqp.data import Data
from piqp.kkt_solver import DenseKKTSolver
from piqp.utils import print_matlab_format


# class TestDenseKKTSolver(unittest.TestCase):
#     def test_dense_kkt_solver_setup(self):
#         # Example data for testing
#         P = np.array([[4.0, 1.0], [1.0, 2.0]])
#         c = np.array([1.0, 1.0])
#         A = np.array([[1.0, 1.0]])
#         b = np.array([1.0])
#         G = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
#         h_u = np.array([0.5, 0.5, 0.0, 0.0])
#         h_l = -np.inf * np.ones_like(h_u)

#         data = Data(P, c, A, b, G, h_u, h_l)
#         solver = DenseKKTSolver(data)

#         solver.update_scalings_and_factor(data, delta=1e-4, x_reg=np.ones(data.n), z_reg=np.ones(data.m))
#         print(solver._AtA.shape)
#         print(solver._kkt_mat)


class TestKKTSystem(unittest.TestCase):
    def test_kkt_system_solve(self):
        # Example data for testing
        P = np.array([[4.0, 1.0], [1.0, 2.0]])
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        G = np.array([[1.0, 0.1], [0.1, 1.0], [-1.0, -0.1], [-0.1, -1.0]])
        h_u = np.array([0.5, 0.5, 1.0, 1.0])
        h_l = -h_u
        x_u = np.array([1.0, 1.0])
        x_l = -x_u

        data = Data(P, c, A, b, G, h_u, h_l, x_u, x_l)

        from piqp.kkt_systems import KKTSystem
        from piqp.results import Variables
        kkt_system = KKTSystem(data)

        delta = 1e-1
        rho = 1e-1
        x_reg = np.ones(data.n)
        z_reg = np.ones(data.m)
        vars = Variables(data.n, data.p, data.m)
        vars.set_random()
        rhs = Variables(data.n, data.p, data.m)
        rhs.set_random()
        lhs = Variables(data.n, data.p, data.m)

        print("The full KKT matrix is: ")

        kkt_mat = kkt_system.kkt_matrix(rho=rho, delta=delta, vars=vars)
        print(kkt_mat.shape)
        print_matlab_format(kkt_mat, name="KKT_Matrix")

        print("The right hand side is: ")
        print_matlab_format(rhs.to_array(), name="RHS")

        sol = kkt_system.kkt_solution(rho=rho, delta=delta, rhs=rhs, vars=vars)
        print("Solution from direct KKT solve:", sol)


        kkt_system.update_scalings_and_factor(data, delta=delta, rho=rho, vars=vars)

        print_matlab_format(kkt_system._kkt_solver._kkt_mat)

        print("The right hand side is: ")
        print_matlab_format(rhs.to_array(), name="RHS")

        lhs = kkt_system.solve(data=data, settings=None, rhs=rhs)
        print("Solution from KKT system solve:", lhs)


        print("\nVerifying the solution, difference is\n: ", lhs - sol)


        import matplotlib.pyplot as plt
        plt.figure()
        plt.spy(kkt_mat)
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.savefig('kkt_matrix_sparsity.png')
        print("KKT matrix sparsity pattern saved to 'kkt_matrix_sparsity.png'")

        self.assertTrue(lhs.allclose(sol, atol=1e-20))


if __name__ == '__main__':
    unittest.main()