import numpy as np
import gmpy2
import itertools
from gmpy2 import mpfr  # float class

# TODO: lower-priority: get speedups from in place ops: *=, /=, +=, -=
# TODO: give MPMatrix access to a MPView.reindex equivalent.
#   No need for the actual reindexing: just re-use the parts of the code
#   which check for types and cast to lists, etc.
# TODO: augment MPMatrix.__getitem__
#   1. Clean the argument (see above) to obtain (int, int)
#       or (list[int], list[int]).
#   2. MPMatrix.__getitem__((list[int], list[int])) should return a view.
# TODO: augment MPMatrix.__setitem__
#   0. Clean the key as above.
#   1. To implement the MPMatrix[list1, list2] case:
#       Verify `(isinstance(value, MPMatrix) and
#               value.shape == (len(list1), len(list2)))`
#   2. Use simple __setitem__ across enumerate(list1) and
#       enumerate(list2) to match up the indices appropriately.


class MPMatrix:
    """Mixed precision matrix class: matrix ops implemented pointwise
    using gmpy2.mpfr"""
    def __init__(self, shape, data):
        """shape (m, n)
        data is a dict of tuple-index 'mpfr' objects"""
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

    def __getitem__(self, key):
        """Syntactic sugar for data read via tuple keys
        
        To access one value, you may use these notations:
        
        0. A[(i,j)] returns the (i,j) entry.

        TODO: 1. A[i, j] returns the (i,j) entry.

        TODO: 2. If A.shape is (m, 1) or (1, n): you may use integer indexing A[i].
        
        TODO: implement views, return them for A[i:, j:] notations.

        TODO: match these implementations to __setitem__
        """

        # Simple indexing A[x]
        return self.data[key]

    def __setitem__(self, key, val):
        """Syntactic sugar for dict write via tuple keys"""
        assert isinstance(val, type(mpfr(0)))
        self.data[key] = val

    def __repr__(self):
        """Printable representation: print each entry,
        using left-justification with space filler.
        
        Will not look pretty if the array is too wide.
        """
        m, n = self.shape
        all_strings = [[self[i, j].__str__() for j in range(n)]
                       for i in range(m)]
        max_len = max(
            [max([len(s) for s in all_strings[i]]) for i in range(m)])
        lines = [
            ''.join([s.ljust(max_len + 1) for s in all_strings[i]])
            for i in range(m)
        ]
        return '\n'.join(lines)

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

    @staticmethod
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

    @staticmethod
    def zeros(m, n):
        """Returns m by n matrix of zeros. No sparsity yet.
        """
        data = dict.fromkeys(itertools.product(range(m), range(n)), mpfr(0))
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
        alpha = self[0, 0]
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
            # Equivalent to self[k:, k] in numpy notation
            data[(i, 0)] = self[k + i, k]
        reflect_me = MPMatrix((length, 1), data)
        v, beta = _house(reflect_me)

        # Now need to update the submatrix B = self.data[k:, k:]
        # via the transformation  B = (I - beta * v * v.T) * B
        scalar = 1 - beta * v.frob_prod(v)
        for i in range(k, m):
            for j in range(k, n):
                self[i, j] *= scalar
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


class MPView(MPMatrix):
    """View of MPMatrix: same underlying data, cf. numpy model."""
    def __init__(self, parent, p_rows, p_cols):
        """Initialize from parent and two lists of parent indices:
        p_rows is parent_rows, p_cols is parent_cols.
        """
        self.parent = parent  # TODO: verify this is memory-cheap
        self.p_rows, self.p_cols = p_rows, p_cols
        new_m = len(self.p_rows)
        new_n = len(self.p_cols)
        self.shape = (new_m, new_n)

        # If B is a matrix view of A, any row/col of B has two indices:
        # the view index in B and the parent index in A.
        # These lists are stored for easy index conversion:
        # self.p_rows[view_row_idx] = parent_row_idx

        # Use this for generating p_rows, p_cols in the caller.
        # m, n = self.parent.shape
        # self.p_rows = list(range(m))[row_slice]
        # self.p_cols = list(range(n))[col_slice]
        return

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

    def slices_to_lists(self, row_slice, col_slice):
        """Converts slices in child to lists in parent.
        This is used when we take a view of a view.
        """
        next_rows = self.p_rows[row_slice]
        next_cols = self.p_cols[col_slice]
        return (next_rows, next_cols)

    def reindex(self, item):
        """Converts view index objects to parent index objects, which
        may be consumed by parent.__getitem__ and parent.__setitem__.
        
        Simple case, item = (int, int): 
            return parent index (int, int)
        Flat case, item = int and view is vector-shaped:
            return parent index (int, int)
        Slice case, item = (slice, slice):
            return coordinate lists (list[int], list[int])
        List case, item = (list[int], list[int]):
            return coordinate lists (list[int], list[int])
        """
        if isinstance(item, int):  # Convert single int to view-coordinates
            flat_dim_idx = list(self.shape).index(1)
            coords = [item, item]
            coords[flat_dim_idx] = 1
            item = tuple(coords)
        assert isinstance(item, tuple)
        assert len(item) == 2
        r, c = item
        if isinstance(r, int):  # simple int pair
            return self.get_parent_idx(item)
        if isinstance(r, slice):  # slice pair
            return self.slices_to_lists(r, c)
        if isinstance(r, list) and isinstance(r[0], int):  # list[int] pair
            new_rows = [self.p_rows[i] for i in r]
            new_cols = [self.p_cols[j] for j in c]
            return new_rows, new_cols

    def __getitem__(self, item):
        """Reindex and use parent's __getitem__"""
        return self.parent[self.reindex(item)]

    def __setitem__(self, item):
        """Reindex and use parent's __setitem__"""
        self.parent.__setitem__(self.reindex(item))


class IndexFetcher:
    """Following numpy.IndexExpression, utility class for index hacks."""
    def __getitem__(self, item):
        return item


def main():
    s = IndexFetcher()
    slice1, slice2 = s[1:, :]
    slice3 = s[:-1:-1]
    x, y = s[2, 3]
    print(x, y)
    print(isinstance(x, slice))
    print(isinstance(slice1, slice))
    print(slice3)
    print(slice3.indices(10))

    return


if __name__ == "__main__":
    main()
