import numpy as np
from scipy.linalg import lu


class PassovnaMatrika():
    def __init__(self, up, diag, low) -> None:
        self.l = len(low)
        self.u = len(up)
        self.m = len(diag)
        self.n = self.l + self.u + 1
        self.up = up
        self.diag = diag
        self.low = low

    def size(self):
        return self.m, self.m

    def getmatrix(self):
        mtx = np.zeros((self.m, self.m))

        np.fill_diagonal(mtx, self.diag)

        if self.u > 0:
            for i in range(self.u):
                vec = self.up[i]
                idx = np.arange(len(vec))
                mtx[idx, idx + i + 1] = vec
        
        if self.l > 0:
            for i in range(self.l):
                vec = self.low[i]
                idx = np.arange(len(vec))
                mtx[idx + i + 1, idx] = vec
        
        return mtx
    
    def dd(self, X):
        D = np.diag(np.abs(X))
        S = np.sum(np.abs(X), axis=1) - D
        return np.all(D > S)

    def nptomatrika(self, x):
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
        x = self.getmatrix()
        
        if self.dd(x):
            l, u = lu(x, permute_l=True)
            print(l)
            return self.nptomatrika(l), self.nptomatrika(u)
        else:
            print('Error. Matrix is not diagonally dominant.')
            return None
            
    def getindex(self, i, j):
        if i == j:
            return self.diag[i]
        elif i < j and j - i < self.n and self.u > 0:
            return self.up[j - i - 1][i]
        elif i > j and i - j < self.n and self.l > 0:
            return self.low[i - j - 1][j]
        else:
            return 0

    def setindex(self, i, j, e):
        if i == j:
            self.diag[i] = e
        elif i < j and j - i < self.n and self.u > 0:
            self.up[j - i - 1][i] = e
        elif i > j and i - j < self.n:
            self.low[i - j - 1][j] = e
        else:
            print('Error. Operation not supported. Either index out of bounds or tt would no longer be a Band matrix.')

    def mat_mul(self, b):
        res = np.zeros(self.size())

        for i in range(self.m):
            for j in range(b.m):
                for k in range(b.m):
                    res[i][j] += self.getindex(i, k) * b[k, j]

        return self.nptomatrika(res)

    def __getitem__(self, index):
        return self.getindex(index[0], index[1])

    def __setitem__(self, index, e):
        self.setindex(index[0], index[1], e)


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
    print(a.mat_mul(b).getmatrix())

if __name__ == '__main__':
    main()
