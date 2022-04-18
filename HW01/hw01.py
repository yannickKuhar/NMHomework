import numpy as np


class PassovnaMatrika():
    def __init__(self, low, diag, up):
        self.l = len(low)
        self.u = len(up)
        self.m = len(diag)
        self.n = self.l + self.u + 1
        self.up = up
        self.diag = diag
        self.low = low

    def size(self):
        """
        Return the size of this matrix.
        :return: Tuple `(m, m)` where m is the length of the diagonal.
        """
        return self.m, self.m

    def getmatrix(self):
        """
        Constructs a numpy matrix from one of the custom formats.
        :return: A numpy array representing our matrix.
        """

        # Initial array full of 0s.
        mtx = np.zeros((self.m, self.m))

        # Fill diagonal with our values
        np.fill_diagonal(mtx, self.diag)

        # If there exist elements above the diagonal fill them to the numpy matrix.
        if self.u > 0:
            for i in range(self.u):
                vec = self.up[i]
                idx = np.arange(len(vec))
                mtx[idx, idx + i + 1] = vec

        # If there exist elements below the diagonal fill them to the numpy matrix.
        if self.l > 0:
            for i in range(self.l):
                vec = self.low[i]
                idx = np.arange(len(vec))
                mtx[idx + i + 1, idx] = vec
        
        return mtx
    
    def dd(self):
        """
        Check if a tridiagonal matrix is diagonally dominant, following the equation:

        https://en.wikipedia.org/wiki/Diagonally_dominant_matrix

        :return: True if it is diagonally dominant anf False if not.
        """

        if abs(self.diag[0]) < abs(self.up[0][0]):
            return False

        for i in range(1, self.n - 1):
            if abs(self.diag[i]) < abs(self.up[0][i] + self.low[0][i]):
                return False

        if abs(self.diag[self.n - 1]) < abs(self.low[0][self.l - 1]):
            return False

        return True

    @staticmethod
    def nptomatrika(x):
        """
        Convert numpy matrix to our format.
        :param x: A numpy matrix.
        :return: The matrix x of type PassovnaMatrika or its descendants.
        """
        upr = []
        d = list(np.diag(x))
        lwr = []

        for i in range(1, len(x)):
            tmp_u = np.diag(x, i)
            tmp_l = np.diag(x, -i)

            if np.any(tmp_u > 0):
                upr.append(list(tmp_u))
            
            if np.any(tmp_l > 0):
                lwr.append(list(tmp_l))

        if len(upr) > 0 and len(lwr) == 0:
            return ZgornjePasovnaMatrika(upr, d)
        elif len(upr) == 0 and len(lwr) > 0:
            return SpodnjePasovnaMatrika(d, lwr)
        else:
            return PassovnaMatrika(upr, d, lwr)

    def lu(self):
        """
        Perform the LU decomposition on this tridiagonal matrix based on the algorithm from the labs.
        :return: Matrices L and U in our format.
        """

        lsp = self.low[0]
        ud = self.diag

        if self.dd():
            for i in range(1, self.n):
                lsp[i - 1] = self.low[0][i - 1] / ud[i - 1]
                ud[i] = ud[i] - lsp[i - 1] * self.up[0][i - 1]

            return SpodnjePasovnaMatrika([lsp], list(np.ones(self.n))), ZgornjePasovnaMatrika(ud, self.up)
        else:
            print('Error. Matrix is not diagonally dominant.')
            return None
            
    def getindex(self, i, j):
        """
        Method that sets element of index `i` and 'j` to `e`. The new indecies are calculated as:
        i = j - i - 1
        j = i,
            for the part above the diagonal and:
        i = i - j - 1
        j = j,
            for the part below.

        :param i: Row index.
        :param j: column index.
        :param e: The value of the element we wish to set.
        :return:
        """
        if i == j:
            return self.diag[i]
        elif i < j and j - i - 1 < self.u and self.u > 0:
            return self.up[j - i - 1][i]
        elif i > j and i - j - 1 < self.l and self.l > 0:
            return self.low[i - j - 1][j]
        else:
            return 0

    def setindex(self, i, j, e):
        """
        Method that sets element of index `i` and 'j` to `e`. The new indecies are calculated as:
        i = j - i - 1
        j = i,
            for the part above the diagonal and:
        i = i - j - 1
        j = j,
            for the part below.

        :param i: Row index.
        :param j: column index.
        :param e: The value of the element we wish to set.
        :return:
        """
        if i == j:
            self.diag[i] = e
        elif i < j and j - i - 1 < self.u and self.u > 0:
            self.up[j - i - 1][i] = e
        elif i > j and i - j - 1 < self.l and self.l > 0:
            self.low[i - j - 1][j] = e
        else:
            print('Error. Operation not supported. Either index out of bounds or tt would no longer be a Band matrix.')
            return None

    def mat_mul(self, b):
        """
        Initialises a numpy array full of zeroes than uses the algorithm that multiplies a tridiagonal
        matrix with vector `b`.
        :param b: Vector `b` of type list.
        :return: The matrix product of this matrix and `b` of type PassovnaMatrika or its descendants.
        """
        res = np.zeros(self.n)

        res[0] = self.diag[0] * b[0] + self.up[0][0] * b[1]

        for i in range(1, self.n - 1):
            res[i] = self.getindex(i, i - 1) * b[i - 1] + self.diag[i] * b[i] + self.getindex(i, i + 1) * b[i + 1]

        res[self.n - 1] = self.low[0][self.n - 2] * b[self.n - 2] + self.diag[self.n - 1] * b[self.n - 1]

        return res

    def mat_div(self, b):
        """
        Implements the algorithm that divides a tridiagonal matrix with vector `b`.
        :param b: Vector `b` of type list.
        :return: The matrix product of this matrix and `b` of type PassovnaMatrika or its descendants.
        """

        x = b[:]
        d = self.diag[:]
        n = len(b)

        for i in range(1, n):

            l = self.low[0][i - 1] / d[i - 1]
            d[i] -= l * self.up[0][i - 1]
            x[i] -= l * x[i - 1]

        x[n - 1] = x[n - 1] / d[n - 1]

        for i in range(n - 2, -1, -1):
            x[i] = (x[i] - x[i + 1] * self.up[0][i]) / d[i]

        return x

    def __getitem__(self, index):
        """
        Magic method that enables the use of `A[i, j]` index format.
        :param index: Tuple of indices (i, j).
        :return: number on the position (i, j).
        """
        return self.getindex(index[0], index[1])

    def __setitem__(self, index, e):
        """
        Magic method that enables the use of `A[i, j] = e` setting format.
        :param index: Tuple of indices (i, j).
        :param e: The value of the element we wish to set.
        :return:
        """
        self.setindex(index[0], index[1], e)

    def __mul__(self, b):
        """
        Magic method that enables the use of `a * b` multiplication format.
        :param b: Vector `b` of type list.
        :return: The matrix product of this matrix and `b` of type PassovnaMatrika or its descendants.
        """
        return self.mat_mul(b)

    def __truediv__(self, b):
        """
        Magic method that enables the use of `a / b` division format.
        :param b: Vector `b` of type list.
        :return: The matrix quotient of this matrix and `b` of type PassovnaMatrika or its descendants.
        """
        return self.mat_div(b)


class ZgornjePasovnaMatrika(PassovnaMatrika):
    def __init__(self, diag, up):
        super().__init__([], diag, up)


class SpodnjePasovnaMatrika(PassovnaMatrika):
    def __init__(self, low, diag):
        super().__init__(low, diag, [])


# def desne_strani(s, d, z, l):
#     n = len(s) + 1
#     m = len(l) + 1
#     b = np.zeros(n * m)
#     b[1:n] -= s
#     b[n:-1] -= d
#     b[(-1 - n + 1) : -1] -= z
#     b[1:n] -= l
#     return b


def main():
    nx = 20
    ny = 20

    funs = [np.sin, lambda y: 0, np.sin, lambda y: 0]

    a, b, c, d = 0, np.pi, 0, np.pi

    Z0 = np.zeros((nx + 2, ny + 2))

    x = np.linspace(a, b, nx + 2)
    y = np.linspace(c, d, ny + 2)

    Z0[:, 1] = list(map(funs[0], x))
    Z0[-1, :] = np.array(list(map(funs[1], y)))
    Z0[:, - 1] = np.array(list(map(funs[2], x)))
    Z0[1, :] = np.array(list(map(funs[3], y)))
    np.fill_diagonal(Z0, -4)

    b = list(np.random.rand(nx + 2))
    Z = PassovnaMatrika.nptomatrika(Z0)

    print(Z / b)


if __name__ == '__main__':
    main()
