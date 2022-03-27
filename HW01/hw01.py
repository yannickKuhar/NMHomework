import numpy as np
from scipy.linalg import lu


class PassovnaMatrika():
    def __init__(self, up, diag, low):
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
    
    def dd(self, X):
        """
        Check if a numpy matrix is diagonally dominant.

        Source: https://stackoverflow.com/questions/43074634/checking-if-a-matrix-is-diagonally-dominant-in-python

        :param X: A numpy matrix constructed from our custom format.
        :return: True if it is diagonally dominant anf False if not.
        """
        D = np.diag(np.abs(X))
        S = np.sum(np.abs(X), axis=1) - D
        return np.all(D > S)

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
        Perform the LU decomposition on this matrix. First we construct a numpy matrix from our own an then use scipy's
        lu() method.
        :return: Matrices L and U in our format.
        """
        x = self.getmatrix()
        
        if self.dd(x):
            l, u = lu(x, permute_l=True)
            # print(l)
            return self.nptomatrika(l), self.nptomatrika(u)
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
        Initialises a numpy array full of zeroes than uses the standard matrix multiplication algorithm tro construct
        the result. Finally we convert the result form numpy array to one of our custom types.
        :param b: Matrix `b` of type PassovnaMatrika or its descendants.
        :return: The matrix product of this matrix and `b` of type PassovnaMatrika or its descendants.
        """
        res = np.zeros(self.size())

        for i in range(self.m):
            for j in range(b.m):
                for k in range(b.m):
                    res[i][j] += self.getindex(i, k) * b[k, j]

        return self.nptomatrika(res)

    def mat_div(self, b):
        """
        Constructs numpy matrices form this matrix and matrix `b` and then uses numpy's matrix division method.
        :param b: Matrix `b` of type PassovnaMatrika or its descendants.
        :return: The matrix product of this matrix and `b` of type PassovnaMatrika or its descendants.
        """
        res = np.linalg.solve(self.getmatrix(), b.getmatrix())
        return self.nptomatrika(res)

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
        :param b: Matrix `b` of type PassovnaMatrika or its descendants.
        :return: The matrix product of this matrix and `b` of type PassovnaMatrika or its descendants.
        """
        return self.mat_mul(b)

    def __truediv__(self, b):
        """
        Magic method that enables the use of `a / b` division format.
        :param b: Matrix `b` of type PassovnaMatrika or its descendants.
        :return: The matrix quotient of this matrix and `b` of type PassovnaMatrika or its descendants.
        """
        return self.mat_div(b)


class ZgornjePasovnaMatrika(PassovnaMatrika):
    def __init__(self, up, diag):
        super().__init__(up, diag, [])


class SpodnjePasovnaMatrika(PassovnaMatrika):
    def __init__(self, diag, low):
        super().__init__([], diag, low)


def desne_strani(s, d, z, l):
    n = len(s) + 1
    m = len(l) + 1
    b = np.zeros(n * m)
    b[1:n] -= s
    b[n:-1] -= d
    b[(-1 - n + 1) : -1] -= z
    b[1:n] -= l
    return b


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

    # print(Z0[:, 1])
    # print(Z0[-1, :])
    # print(Z0[:, - 1])
    # print(Z0[1, :])

    Z = PassovnaMatrika.nptomatrika(Z0)
    b = desne_strani(Z0[2:-1-1, 1], Z0[-1, 2:-1-1],Z0[2:-1-1, -1], Z0[1, 2:-1-1])
    print(Z0[2:-1-1, 1])


if __name__ == '__main__':
    main()
