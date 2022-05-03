import numpy as np

class RazprsenaMatrika():
    def __init__(self, V, I):
        self.V = V
        self.I = I

    def getindex(self, i, j):
        return self.V[i][self.I[i].index(j)] if j in self.I[i] else 0

    def __getitem__(self, index):
        """
        Magic method that enables the use of `A[i, j]` index format.
        :param index: Tuple of indices (i, j).
        :return: number on the position (i, j).
        """
        return self.getindex(index[0], index[1])

    def setindex(self, i, j, e):
        if j in self.I[i]:
            self.V[i][self.I[i].index(j)] = e
        else:
            idx = -1
            for k in self.I[i]:
                if j > k:
                    idx = i
                    break

            self.I[i].insert(idx, j)
            self.V[i].insert(idx, e)

    def __setitem__(self, index, e):
        """
        Magic method that enables the use of `A[i, j] = e` setting format.
        :param index: Tuple of indices (i, j).
        :param e: The value of the element we wish to set.
        :return:
        """
        self.setindex(index[0], index[1], e)

    def mat_mul(self, b):
        res = [0] * len(b)

        for k in range(len(self.I)):
            for i in self.I[k]:
                res[k] += b[i] * self.V[k][self.I[k].index(i)]

        return np.array(res)

    def firstindex(self, i):
        return self.I[i][0]

    def lastindex(self, i):
        return self.I[i][-1]

    def __mul__(self, b):
        """
        Magic method that enables the use of `a * b` multiplication format.
        :param b: Vector `b` of type list.
        :return: The matrix product of this matrix and `b` of type RazprsenaMatrika.
        """
        return self.mat_mul(b)


def conj_grad(A, b, x):
    r = b - A * x
    p = r
    rsold = r.T @ r
    
    for _ in b:
        Ap = A * p
        alpha = rsold / (p.T @ Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        rsnew = r.T @ r
        
        if rsnew < 1e-10:
            return x

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def main():
    A = np.array([[2, 1, 0, 0], [0, 2, 0, 0], [0, 1, 3, 0], [0, 0, 0, 1]])

    print(A)

    B = np.array([3, 3, 3, 3])
    print(np.linalg.solve(A, B))


    a = RazprsenaMatrika([[2, 1], [2], [1, 3], [1]], [[0, 1], [1], [1, 2], [3]])
    b = [3, 3, 3, 3]

    # print(a * b)

    x = np.array([1, 1, 1, 1])
    x = conj_grad(a, b, x)
    print(x)


if __name__ == '__main__':
    main()
