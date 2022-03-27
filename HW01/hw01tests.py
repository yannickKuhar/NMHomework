import unittest
import numpy as np
from hw01 import PassovnaMatrika, ZgornjePasovnaMatrika, SpodnjePasovnaMatrika

a = PassovnaMatrika([[1, 1]], [5, 5, 5], [[1, 1]])

A = np.array([[5, 1, 0],
              [1, 5, 1],
              [0, 1, 5]])


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

    def


if __name__ == '__main__':
    unittest.main()
