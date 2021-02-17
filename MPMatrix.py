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

    def __setitem__(self, key, val):
        """Syntactic sugar for dict write via tuple keys"""
        assert isinstance(val, type(mpfr(0)))  # test
        self.data[key] = val

    def get_rows(self, k, row_count):
        """Returns an MPMatrix given by row_count rows, starting from k.
        Equivalent to A[k:k+row_count] in numpy.
        If row_count = -1, it returns all rows from k onward,
        equivalent to A[k:] in numpy."""
        data = dict()
        n, m = self.shape

        if row_count == -1:
            row_count = n - k
        for j in range(m):
            for i in range(row_count):
                data[(i, j)] = self.data[(k + i, j)]
        return MPMatrix((row_count, m), data)

    def get_cols(self, k, col_count):
        """Returns an MPMatrix given by col_count columns, starting from k.
        Equivalent to A[:, k:k+col_count] in numpy.
        If col_count = -1, it returns all cols from k onward,
        equivalent to A[:, k:] in numpy."""
        data = dict()
        n, m = self.shape

        if col_count == -1:
            col_count = m - k
        for j in range(col_count):
            for i in range(n):
                data[(i, j)] = self.data[(i, k + j)]
        return MPMatrix((n, col_count), data)

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

    def copy(self):
        """Returns a copy of itself."""
        data = self.data.copy()
        return MPMatrix(self.shape, data)

    def scale(self, scalar):
        """A.scale(c) returns c*A, pointwise multiplication.
        For precise scaling, provide the scalar as a string, e.g. "1.233".
        """
        n, m = self.shape
        for i in range(n):
            for j in range(m):
                self[(i, j)] *= mpfr(scalar)
        return self

    def ptwise(self, f):
        """A.ptwise(f) returns an MPMatrix with f applied to each entry of A.
        The function f should be well-defined on the mpfr type."""
        n, m = self.shape
        for i in range(n):
            for j in range(m):
                self[(i, j)] = f(self[(i, j)])
        return self

    def frob_prod(self, B):
        """A.frob_prod(B) returns the Frobenius inner product <A,B>.
        B is also an MPMatrix. Not implemented for complex types."""
        n, m = self.shape
        k, r = B.shape
        assert (n == k
                and m == r), ("Distinct shapes ({}, {}) and ({}, {})".format(
                    n, m, k, r))
        sum_ = 0
        for i in range(n):
            for j in range(m):
                sum_ += self[(i, j)] * B[(i, j)]
        return sum_

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

    def _house(self):
        """Householder reflection, defined for column vector shapes x=(m,1).
        Returns (v, beta), where v is a column vector v with v[0]=1,
        and the matrix P = I - beta * v * v.T gives the Householder
        transformation satisfying: Px = norm(x)*e_1.
        
        Context: recall that a Householder transformation is given by an
        m-by-m matrix of the form: P = I - beta * v * v.T
        It reflects over v, the Householder vector.
        
        The matrix is never explicitly formed.
        Further info: algorithm 5.1.1, pg 236, Matrix Computations (4th ed).
        
        To efficiently apply the transformation to a matrix:

        PA = A - (beta * v) (v.T * A)
        AP = A - (A*v) (B*v).T
        """
        m, n = self.shape
        assert n == 1, "need shape (m,1); have ({}, {})".format(m, n)
        alpha = self.data(0, 0)
        y = self.drop_row(0)
        sigma = y.frob_prod(y)  #norm squared

        if sigma == 0 and alpha >= 0:
            return self, 0
        elif sigma == 0 and alpha < 0:
            return self, -2
        else:
            v = self.copy()
            mu = gmpy2.sqrt(alpha**2 + sigma)  # always positive
            if alpha <= 0:
                new_v0 = alpha - mu
            else:
                new_v0 = -sigma / (alpha + mu)
            beta = 2 * new_v0**2 / (sigma + new_v0**2)
            v[(0, 0)] = new_v0
            v = v.scale(1 / new_v0)
            return v, beta

    def _house_minor_update(self, k):
        """Utility function used for algorithm 5.2.1, pg 273,
        Matrix Computation 4th ed.
        
        Mutates self by clearing the kth column.
        """
        m, n = self.shape

        # Copy the column vector in order to call _house(column).
        data = dict()
        length = m - k
        for i in range(length):
            # Equivalent to self.data[k:, k] in numpy notation
            data[(i, 0)] = self.data[(k + i, k)]
        reflect_me = MPMatrix((length, 1), data)
        v, beta = _house(reflect_me)

        # Now need to update the submatrix B = self.data[k:, k:]
        # via the transformation  B = (I - beta * v * v.T) * B
        scalar = 1 - beta * v.frob_prod(v)
        for i in range(k, m):
            for j in range(k, n):
                self.data[(i, j)] *= scalar
                # Could we optimize this by storing scalars for n minors?
        return

    def QR(self):
        """Perform QR factorization with householder reflections.
        O(n^3), stable."""
        m, n = self.shape
        for j in range(n):
            _house_minor_update(self)
            if j < m:
                #TODO: A(j+1:m, j) = v(2:m-j+1)
                pass
        return


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
