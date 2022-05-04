import numpy as np

class RazprsenaMatrika():
    def __init__(self, V, I):
        self.V = V
        self.I = I

    def getindex(self, i, j):
        """
        Method that gets the element on index `i` and 'j`.
        :param i: Row index.
        :param j: column index.
        :return: The element on index (i, j), if it exists, else 0.
        """
        return self.V[i][self.I[i].index(j)] if j in self.I[i] else 0

    def __getitem__(self, index):
        """
        Magic method that enables the use of `A[i, j]` index format.
        :param index: Tuple of indices (i, j).
        :return: number on the position (i, j).
        """
        return self.getindex(index[0], index[1])

    def setindex(self, i, j, e):
        """
        Sets the the element on index (i, j) to element `i`. If the index
        already exists in matrix I the vale of the elemenit is replaced if 
        not it is inserted into its proper place.
        """
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
        """
        Implements the algorithm that multiplies a sparse matrix with vector `b`.
        :param b: Vector `b` of type list.
        :return: The matrix product of this matrix and `b`.
        """
        res = [0] * len(b)

        for k in range(len(self.I)):
            for i in self.I[k]:
                res[k] += b[i] * self.V[k][self.I[k].index(i)]

        return np.array(res)

    def firstindex(self, i):
        """
        Returns the 1st index in row `i`.
        :param i: The index of a row.
        :return: The 1st index of ith row in matrx I. 
        """
        return self.I[i][0]

    def lastindex(self, i):
        """
        Returns the last index in row `i`.
        :param i: The index of a row.
        :return: The last index of ith row in matrx I. 
        """
        return self.I[i][-1]

    def __mul__(self, b):
        """
        Magic method that enables the use of `a * b` multiplication format.
        :param b: Vector `b` of type list.
        :return: The matrix product of this matrix and `b` of type RazprsenaMatrika.
        """
        return self.mat_mul(b)


def conj_grad(A, b, x):
    """
    Implements the Conjugate gradient method. Vector `r` is the difference between te intial guess and the solution
    with this method iteratively the gram schimd method is applied on vector `r` so it becomes more and more orthogonal
    while at the same time the movements are applied to `x`, our intial guess. At the end `r` is as close to 0 as can be
    and `x` as close to the solution.
    :param A: The matrix in our equation system Ax = b. 
    :param b: The solution vector.
    :param x: The intitial guess.
    :return x: The optimised vector `x`.
    """
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
    
    B = np.array([3, 3, 3, 3])
    print(np.linalg.solve(A, B))


    a = RazprsenaMatrika([[2, 1], [2], [1, 3], [1]], [[0, 1], [1], [1, 2], [3]])
    b = [3, 3, 3, 3]

    x = np.array([1, 1, 1, 1])
    x = conj_grad(a, b, x)
    print(x)


if __name__ == '__main__':
    main()
