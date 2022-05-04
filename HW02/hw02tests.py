import unittest
import numpy as np
from hw02 import RazprsenaMatrika

A = np.array([[2, 1, 0, 0], [0, 2, 0, 0], [0, 1, 3, 0], [0, 0, 0, 1]])

a = RazprsenaMatrika([[2, 1], [2], [1, 3], [1]], [[0, 1], [1], [1, 2], [3]])

c = [3, 3, 3, 3]


class TestMatrix(unittest.TestCase):
    """
    A class that runs a few basic tests.
    """

    def test_get_index(self):
        for i in range(len(A)):
            for j in range(len(A[i])):
                self.assertEqual(a[i, j], A[i][j])

    def test_set_index(self):
        a[2, 2] = 7
        self.assertEqual(a[2, 2], 7)
        a[2, 2] = 5
        self.assertEqual(a[2, 2], 5)

    def test_mat_mul(self):
        self.assertEqual(np.all((a * c) == (A.dot(c))), True)


if __name__ == '__main__':
    unittest.main()
