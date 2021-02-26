import numpy as np
import gmpy2
import itertools
from gmpy2 import mpfr  # float class

# TODO: lower-priority: get speedups from in place ops: *=, /=, +=, -=
# TODO: abstract the index-cleaning part of __getitem__ and __setitem__
# on MPMatrix and MPView


def import_array(A):
    """A is, e.g., numpy array"""
    d = len(A.shape)
    assert d == 2, "Cannot import {} dimension array, need 2".format(d)
    m, n = A.shape
    data = dict()
    for i in range(m):
        for j in range(n):
            data[i, j] = mpfr(A[(i, j)])
    return MPMatrix((m, n), data)


def zeros(m, n):
    """Returns m by n matrix of zeros. No sparsity yet.
    """
    data = dict.fromkeys(itertools.product(range(m), range(n)), mpfr(0))
    return MPMatrix((m, n), data)


def eye(m):
    """Return m by m identity MPMatrix"""
    data = dict()
    for i, j in range(itertools.product(range(m), range(m))):
        data[i, j] = mpfr(i == j)
    return MPMatrix((m, m), data)


class MPMatrix:
    """Mixed precision matrix class: matrix ops implemented pointwise
    using gmpy2.mpfr"""
    def __init__(self, shape, data):
        """shape (m, n)
        data is a dict of tuple-index 'mpfr' objects
        """
        self.shape = shape
        self.data = data

    def __add__(self, B):
        """B is another MPMatrix, returns A+B using global ctx"""
        m, n = self.shape
        k, r = B.shape
        assert (m == k
                and n == r), ("Cannot add shapes ({}, {}) and ({}, {})".format(
                    m, n, k, r))
        sum_ = dict()
        for i in range(m):
            for j in range(n):
                sum_[i, j] = self[i, j] + B[i, j]
        return MPMatrix((m, n), sum_)

    def __mul__(self, B):
        """B is another MPMatrix, returns A*B using global ctx"""
        m, n = self.shape
        n_, r = B.shape
        assert n == n_, ("Cannot multiply shapes "
                         "({}, {}) and ({}, {})".format(m, n, n_, r))
        mul_ = dict()
        # compute A_ik = sum_j A_ij*B_jk
        for i in range(m):
            for k in range(r):
                prod = mpfr(0)
                for j in range(n):
                    prod += self[i, j] * B[j, k]
                mul_[i, k] = prod
        return MPMatrix((m, r), mul_)

    def _cleankey(self, key):
        """Cleans __getitem_ input.
        If key is an int, it will be interpreted as a vector index if possible.
            Ex: v is a row vector, v[0] will return the 0th entry of v.
            If it cannot be interpreted as a vector index,
                it is interpreted as a row index.
        
        If key is a slice, it is converted to a list.

        If key is a list, it will be interpreted as a list of vector indices
            if possible. Otherwise it will be interpreted as a list of
            row indices.

        If key is a tuple:
            Convert each entry to a list:
                a int -> [a] list
                a:b:c slice -> [a, a+c, ... a+nc] list
                    (by applying the slice to the row or column indices)
        
        Always returns (row_indices, column_indices) index of the given key.
        """
        if isinstance(key, tuple) and len(key) == 2 and isinstance(
                key[0], list) and isinstance(key[1], list):
            return key

        if isinstance(key, MPView):
            key = (key.p_rows, key.p_cols)
            return key

        m, n = self.shape

        if isinstance(key, int) or isinstance(key, slice) or isinstance(
                key, list):  # One index given, check if vector shaped.
            if m == 1:  # row vector case
                key = ([0], key)
            elif n == 1:  # col vector case
                key = (key, [0])
            else:  # if not vector-shaped, all columns are implicitly indexed
                key = (key, list(range(n)))
        row_key, col_key = key

        if isinstance(row_key, slice):
            row_key = list(range(m))[row_key]
        elif isinstance(row_key, int):
            row_key = [row_key]

        if isinstance(col_key, slice):
            col_key = list(range(n))[col_key]
        elif isinstance(col_key, int):
            col_key = [col_key]

        return (row_key, col_key)

    def __getitem__(self, key):
        """Syntactic sugar for data read via tuple keys
        
        To access one value, you may use these notations:
        
        0. A[(i,j)] returns the (i,j) entry.

        1. A[i, j] returns the (i,j) entry.

        2. If A.shape is (m, 1) or (1, n): you may use integer indexing A[i].
        
        To access multiple values: key should specify a view.
        The key type must be a length-2 tuple, where each entry can be
        a slice, index list, or int.

        Example: A[[2], [4,5,6]] will attempt to return a view of the
        submatrix given by row 2, columns 4, 5, and 6.

        A[2, [4,5,6]] and A[2, 4:7] are valid ways to obtain the same submatrix.
        """
        rows, cols = self._cleankey(key)
        if len(rows) == len(cols) == 1:  #simple index case
            i, j = rows[0], cols[0]
            return self.data[i, j]
        return MPView(self, rows, cols)

    def __setitem__(self, key, val):
        """
        Cases:
        1. Key is a simple index, val is a simple mpfr.
        1a. Key is two ints: basic case.
        1b. Key is one int: coalesce to two ints.

        2. Key indexes a submatrix. Val is another MPMatrix of appropriate size.
        First, replace slices with index lists. Replace ints with index lists.
        2a. Key is two lists: good.
        2b. Key is a view: get p_rows, p_cols, case 2a.
        
        If Key is a view: we simply need to iterate and setitem.
        """
        rows, cols = self._cleankey(key)
        if len(rows) == len(cols) == 1:  #simple index case
            i, j = rows[0], cols[0]
            assert isinstance(val, type(mpfr(0)))
            self.data[i, j] = val
            return

        assert (len(rows), len(cols)) == val.shape

        # Local data indices: a, b.
        # View indices: i, j
        for i, a in enumerate(rows):
            for j, b in enumerate(cols):
                datum = val[i, j]
                assert isinstance(datum, type(mpfr(0)))
                self.data[a, b] = val[i, j]
        return

    def __repr__(self):
        """Printable representation: print each entry,
        using left-justification with space filler.
        
        Will not look pretty if the array is too wide.
        """
        m, n = self.shape
        if 0 in (m, n):
            return "Empty or invalid MPMatrix of shape {}".format(self.shape)

        all_strings = [[self[i, j].__str__() for j in range(n)]
                       for i in range(m)]
        max_len = max(
            [max([len(s) for s in all_strings[i]]) for i in range(m)])
        lines = [
            ''.join([s.ljust(max_len + 1) for s in all_strings[i]])
            for i in range(m)
        ]
        return '\n'.join(lines)

    def copy(self):
        """Returns a copy of itself."""
        data = self.data.copy()
        return MPMatrix(self.shape, data)

    def scale(self, scalar):
        """A.scale(c) returns c*A, pointwise multiplication.
        For precise scaling, provide the scalar as a string, e.g. "1.233".
        """
        m, n = self.shape
        for i in range(m):
            for j in range(n):
                self[(i, j)] *= mpfr(scalar)
        return self

    def ptwise(self, f):
        """A.ptwise(f) returns an MPMatrix with f applied to each entry of A.
        The function f should be well-defined on the mpfr type."""
        m, n = self.shape
        for i in range(m):
            for j in range(n):
                self[(i, j)] = f(self[(i, j)])
        return self

    def frob_prod(self, B):
        """A.frob_prod(B) returns the Frobenius inner product <A,B>.
        B is also an MPMatrix. Not implemented for complex types."""
        m, n = self.shape
        k, r = B.shape
        assert (m == k
                and n == r), ("Distinct shapes ({}, {}) and ({}, {})".format(
                    m, n, k, r))
        sum_ = 0
        for i in range(m):
            for j in range(n):
                sum_ += self[(i, j)] * B[(i, j)]
        return sum_

    def T(self):
        m, n = self.shape
        return MPView(self, list(range(m)), list(range(n)), is_transpose=True)


# TODO: this implementation depends on deprecated functions, fix.

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
        alpha = self[0, 0]
        y = self[1:]
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

    def QR(self):
        """Perform QR factorization with householder reflections.
        Assumes m >= n. O(n^3), stable.
        """
        m, n = self.shape
        R = self.copy()
        Q = eye(m)

        for j in range(n):
            v, beta = R[j:, j]._house()
            H = eye(m)
            prod = v.frob_product(v)
            H[j:, j:] -= beta * prod  # A[j:, j:] = (I - beta*v*v.T)*A[j:, j:]
            if j < m - 1:
                H[j + 1:, j] = v[2:m - j + 1]  # A[j+1:, j] = v[2:m-j+1]
            R = H * A
            Q = H * Q
        return Q[:n].T, R[:n]


class MPView(MPMatrix):
    """View of MPMatrix: same underlying data, cf. numpy model."""
    def __init__(self, parent, p_rows, p_cols, is_transpose=False):
        """Initialize from parent and two lists of parent indices:
        p_rows is parent_rows, p_cols is parent_cols.
        
        Ex: If A.shape = (5,5), MPView(A, [2,3,4], [2,3,4]) will initialize
        an MPView of shape (3,3), looking at the 3x3 bottom-right submatrix.
        
        TODO: verify transpose is robust
        """
        self.parent = parent
        self.p_rows, self.p_cols = p_rows, p_cols
        self.is_transpose = is_transpose
        new_m = len(self.p_rows)
        new_n = len(self.p_cols)
        self.shape = (new_m, new_n)
        if new_m == 0 or new_n == 0:
            raise IndexError('Invalid shape: {},{}'.format(new_m, new_n))

        # If B is a matrix view of A, any row/col of B has two indices:
        # the view index in B and the parent index in A.
        # These lists are stored for easy index conversion:
        # self.p_rows[view_row_idx] = parent_row_idx

        # Use this for generating p_rows, p_cols in the caller.
        # m, n = self.parent.shape
        # self.p_rows = list(range(m))[row_slice]
        # self.p_cols = list(range(n))[col_slice]
        return

    def copy(self):
        """Returns a full-fledged MPMatrix with the same indices as the view."""
        data = dict()
        m, n = self.shape
        for i in range(m):
            for j in range(n):
                data[i, j] = self[i, j]
        return MPMatrix(self.shape, data)

    def get_view_idx(self, parent_idx):
        """Index conversion from parent index (i,j) to child index (a,b).
        An out-of-bounds index raises an exception.
        """
        a = self.p_rows.index(i)
        b = self.p_cols.index(j)
        return (a, b)

    def get_parent_idx(self, view_idx):
        """Index conversion from child index (a,b) to parent index (i,j).
        An out-of-bounds index raises an exception.
        """
        a, b = view_idx
        R, C = self.shape
        i = self.p_rows[a]
        j = self.p_cols[b]
        return (i, j)

    def __repr__(self):
        """copied from parent but swapping n and m"""
        if self.is_transpose:
            n, m = self.shape
        else:
            m, n = self.shape
        if 0 in (m, n):
            return "Empty or invalid MPMatrix of shape {}".format(self.shape)

        all_strings = [[self[i, j].__str__() for j in range(n)]
                       for i in range(m)]
        max_len = max(
            [max([len(s) for s in all_strings[i]]) for i in range(m)])
        lines = [
            ''.join([s.ljust(max_len + 1) for s in all_strings[i]])
            for i in range(m)
        ]
        return '\n'.join(lines)

    def reindex(self, item):
        """Converts view index objects to parent index objects, which
        may be consumed by parent.__getitem__ and parent.__setitem__.
        
        This also handles transpose reindexing, where the parent holds
        the non-transposed data.
        """
        item = self._cleankey(item)
        if self.is_transpose:
            c, r = item
        else:
            r, c = item

        if isinstance(r, list) and isinstance(c, list):
            new_rows = [self.p_rows[i] for i in r]
            new_cols = [self.p_cols[j] for j in c]
            return new_rows, new_cols
        else:
            raise

    def __getitem__(self, item):
        """Reindex and use parent's __getitem__"""
        index = self.reindex(item)
        return self.parent[index]

    def __setitem__(self, item, value):
        """Reindex and use parent's __setitem__"""
        index = self.reindex(item)
        self.parent.__setitem__(index, value)


class IndexFetcher:
    """Following numpy.IndexExpression, utility class for index hacks."""
    def __getitem__(self, item):
        return item


def main():
    return


if __name__ == "__main__":
    main()
