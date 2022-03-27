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

    def nptomatrika(self, x):
        """
        Convert numpy matrix to our format.
        :param x: A numpy matrix.
        :return: The matrix x of type PassovnaMatrika or its descendants.
        """
        upr = []
        d = list(np.diag(x))
        lwr = []

        for i in range(1, self.n):
            tmp_u = np.diag(x, i)
            tmp_l = np.diag(x, -i)

            if np.all(tmp_u > 0):
                upr.append(list(tmp_u))
            
            if np.all(tmp_l > 0):
                lwr.append(list(tmp_l))

        if self.u > 0 and self.l == 0:
            return ZgornjePasovnaMatrika(upr, d)
        elif self.u == 0 and self.l > 0:
            return SpodnjePasovnaMatrika(d, lwr)
        else:
            return PassovnaMatrika(upr, d, lwr)

    def lu(self):
        """
        PErform the LU decomposition on this matrix. First we construct a numpy matrix from our own an then use scipy's
        lu() method.
        :return: Matrices L and U in our format.
        """
        x = self.getmatrix()
        
        if self.dd(x):
            l, u = lu(x, permute_l=True)
            print(l)
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
        elif i < j and j - i < self.n and self.u > 0:
            return self.up[j - i - 1][i]
        elif i > j and i - j < self.n and self.l > 0:
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
        elif i < j and j - i < self.n and self.u > 0:
            self.up[j - i - 1][i] = e
        elif i > j and i - j < self.n:
            self.low[i - j - 1][j] = e
        else:
            print('Error. Operation not supported. Either index out of bounds or tt would no longer be a Band matrix.')

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
        res = np.divide(self.getmatrix(), b.getmatrix)
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


def main():
    a = ZgornjePasovnaMatrika([[4, 5]], [1,2,3])
    b = SpodnjePasovnaMatrika([1,2,3],[[4, 5]])

    c = a.getmatrix()
    d = b.getmatrix()

    print(c)
    print(d)
    print('////////////')
    print(c @ d)
    print((a * b).getmatrix())


if __name__ == '__main__':
    main()
