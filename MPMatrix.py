import numpy as np
import gmpy2
from gmpy2 import mpfr  # float class

#   Attributes:
#       * shape
#       * data accessible as tuple-indexed dictionary
#       ! precision doesn't live at this structural level.
#   Methods:
#       # basic matrix ops using global context
#       * add two matrices
#       * multiply two matrices
#       * scalar multiplication by an mpfr?
#
# TODO: scalar mult


class MPMatrix:
    def __init__(self, shape, data):
        """shape (n,m)
        data is a dict of tuple-index 'mpfr' objects"""
        self.shape = shape
        self.data = data

    def __add__(self, B):
        """B is another MPMatrix, returns A+B using global ctx"""
        n, m = self.shape
        k, r = B.shape
        assert (n == k
                and m == r), ("Cannot add shapes ({}, {}) and ({}, {})".format(
                    n, m, k, r))
        sum_ = dict()
        for i in range(n):
            for j in range(m):
                sum_[(i, j)] = self.data[(i, j)] + B.data[(i, j)]
        return MPMatrix((n, m), sum_)

    def __mul__(self, B):
        """B is another MPMatrix, returns A*B using global ctx"""
        n, m = self.shape
        m_, r = B.shape
        assert m == m_, ("Cannot multiply shapes "
                         "({}, {}) and ({}, {})".format(n, m, m_, r))
        mul_ = dict()
        # compute A_ik = sum_j A_ij*B_jk
        for i in range(n):
            for k in range(r):
                prod = mpfr(0)
                for j in range(m):
                    prod += self.data[(i, j)] * B.data[(j, k)]
                mul_[(i, k)] = prod
        return MPMatrix((n, r), mul_)

    def __getitem__(self, key):
        """Syntactic sugar for data read via tuple keys"""
        return self.data[key]

    def get_row(self, k):
        """Returns an MPMatrix given by row k"""
        data = dict()
        n, m = self.shape
        for j in range(m):
            data[(0, j)] = self.data[(k, j)]
        return MPMatrix((1, m), data)

    def drop_row(self, k):
        """Returns an MPMatrix given by dropping row k and reindexing"""
        data = self.data.copy()
        n, m = self.shape
        for i in range(n):
            if i < n:
                pass
            elif i == n:  # delete all keys
                for j in range(m):
                    del data[(i, j)]
            else:  # reindex all keys
                for j in range(m):
                    data[(i - 1, j)] = data.pop((i, j))
        return MPMatrix((n - 1, m), data)

    def __setitem__(self, key, val):
        """Syntactic sugar for dict write via tuple keys"""
        assert isinstance(val, type(mpfr(0)))  # test
        self.data[key] = val

    def scale(self, scalar):
        """A.scale(c) returns c*A, pointwise multiplication.
        For precise scaling, provide the scalar as a string, e.g. "1.233".
        """
        n, m = self.shape
        for i in range(n):
            for j in range(m):
                self[(i, j)] *= mpfr(scalar)
        return self

    @staticmethod
    def import_array(A):
        """A is, e.g., numpy array"""
        d = len(A.shape)
        assert d == 2, "Cannot import {} dimension array, need 2".format(d)
        n, m = A.shape
        data = dict()
        for i in range(n):
            for j in range(m):
                data[(i, j)] = mpfr(A[(i, j)])
        return MPMatrix((n, m), data)

    @staticmethod
    def zeros(m, n):
        """Returns m by n matrix of zeros. No sparsity yet."""
        data = dict()
        for i in range(m):
            for j in range(n):
                data[(i, j)] = mpfr(0)
        return MPMatrix((m, n), data)

    def QR(A):
        """Perform QR factorization"""


def matrix_test1():
    A = np.array([[1., 2], [100, 2]])
    B = np.array([[0.01, 0.02], [0.3, 0.001]])
    A_ = MPMatrix.import_array(A)
    B_ = MPMatrix.import_array(B)
    C = A_ + B_
    D = A_ * B_
    gmpy2.get_context().precision = 8
    print((A + B)[(0, 0)], C.data[(0, 0)])
    print((A + B)[(0, 0)], C[(0, 0)])
    print((np.matmul(A, B))[(0, 0)], D[(0, 0)])
    return (C, D)  #hand tested: this works as anticipated


def main():
    C, D = matrix_test1()


if __name__ == "__main__":
    main()
