import unittest
import numpy as np
from scipy.linalg import lu
from hw01 import PassovnaMatrika, ZgornjePasovnaMatrika, SpodnjePasovnaMatrika


a = PassovnaMatrika([[1, 1]], [5, 5, 5], [[1, 1]])
b = PassovnaMatrika([[5, 5]], [1, 1, 1], [[5, 5]])

A = np.array([[5, 1, 0],
              [1, 5, 1],
              [0, 1, 5]])

B = np.array([[1, 5, 0],
              [5, 1, 5],
              [0, 5, 1]])


class TestMatrix(unittest.TestCase):

    def test_get_index(self):
        for i in range(a.m):
            for j in range(a.m):
                self.assertEqual(a[i, j], A[i, j])

    def test_set_index(self):
        a[2, 2] = 7
        self.assertEqual(a[2, 2], 7)
        a[2, 2] = 5
        self.assertEqual(a[2, 2], 5)

    def test_get_matrix(self):
        self.assertEqual(np.all(a.getmatrix() == A), True)

    def test_lu(self):
        l, u = a.lu()
        lt, ut = lu(A, permute_l=True)

        self.assertEqual(np.all(l.getmatrix() == lt), True)
        self.assertEqual(np.all(u.getmatrix() == ut), True)

    def test_lu2(self):
        result = b.lu()
        self.assertEqual(result, None)

    def test_mat_mul(self):
        self.assertEqual(np.all((a * b).getmatrix() == (A @ B)), True)

    def test_mat_div(self):
        self.assertEqual(np.all((a / a).getmatrix() == (np.linalg.solve(A, A))), True)


if __name__ == '__main__':
    unittest.main()
