import unittest
import numpy as np
import gmpy2
from gmpy2 import mpfr  # float class
import itertools
import random
from MPMatrix import MPMatrix

# See: https://gmpy2.readthedocs.io/en/latest/mpfr.html
# Note: contexts are not thread-safe.
# Modifying a context applies to all threads.
# gmpy2.context() initializes a new context.
# gmpy2.get_context() gets a reference to current context. This is mutable.
# gmpy2.set_context(ctx) sets active context.
# gmpy2.local_context([ctx], **kwargs) saves current context,
#   creates a new one (or uses passed context), applies **kwargs.
# Can simulate arbitrary floats by choice of emin, emax, precision.


def _ptwise_vals_equal(mp_val, np_val, epsilon):
    """Pointwise value comparison"""
    # TODO change to relative error bound or find appropriate absolute bound
    ptwise_diff = abs(mp_val - mpfr(np_val))
    return ptwise_diff < epsilon


def _assert_mp_equals_np(mp_array, np_array):
    """Compares an MPMatrix and np array, checking for equality.
    Returns an equality bool and an error string."""
    m, n = np_array.shape
    coordinates = itertools.product(range(m), range(n))
    P = gmpy2.get_context().precision
    epsilon = mpfr("0b" + "0" * P + "1")
    for coord in coordinates:
        mp_val = mp_array[coord]
        np_val = np_array[coord]
        if not _ptwise_vals_equal(mp_val, np_val, epsilon):
            return (False, "Distinct values mp: {}, np: {} at coord {}".format(
                mp_val, np_val, coord))
    return (True, "")


# Verifying basic behaviors of gmpy2: context changes.
class prec1(unittest.TestCase):
    def test(self):
        with gmpy2.local_context() as ctx1:  # sandbox context
            ctx1.precision = 4
            x = mpfr(1) / 7
            with gmpy2.local_context() as ctx2:
                # base 10: 0.141
                self.assertEqual(x.digits(), ('141', 0, 4))
                # base 2: 0.001001
                self.assertEqual(x.digits(2), ('1001', -2, 4))
                ctx2.precision += 20
                # changing context precision does not impact existing values
                self.assertEqual(x.digits(2), ('1001', -2, 4))
            # returning to distinct context does not impact existing values
            self.assertEqual(x.digits(2), ('1001', -2, 4))
        return


# Verify import from numpy array works on a simple test case
class ImportNP(unittest.TestCase):
    def test(self):
        n = random.randint(1, 3)
        A = 100 * np.random.rand(n, n)
        A_ = MPMatrix.import_array(A)
        equality, log = _assert_mp_equals_np(A_, A)
        self.assertTrue(equality, msg=log)


class AddMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(1, 3), random.randint(1, 3)
        A, B = 100 * np.random.rand(m, n), 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        B_ = MPMatrix.import_array(B)
        equality, log = _assert_mp_equals_np(A_ + B_, A + B)
        self.assertTrue(equality, msg=log)


class MatMulMP(unittest.TestCase):
    def test(self):
        # Test multiplying (m,k)*(k,n) matrices
        m, n = random.randint(1, 3), random.randint(1, 3)
        k = random.randint(1, 3)
        A, B = 100 * np.random.rand(m, k), 100 * np.random.rand(k, n)
        A_ = MPMatrix.import_array(A)
        B_ = MPMatrix.import_array(B)
        AB_np = np.matmul(A, B)
        AB_mp = A_ * B_
        equality, log = _assert_mp_equals_np(AB_mp, AB_np)
        self.assertTrue(equality, msg=log)


class ScaleFloatMP(unittest.TestCase):
    def test(self):
        # Test scaling
        m, n = random.randint(1, 3), random.randint(1, 3)
        A = 100 * np.random.rand(m, n)
        c = random.random()  # float in [0,1)
        A_ = MPMatrix.import_array(A)
        cA_np = c * A
        cA_mp = A_.scale(c)
        equality, log = _assert_mp_equals_np(cA_mp, cA_np)
        self.assertTrue(equality, msg=log)


class ScaleStringMP(unittest.TestCase):
    def test(self):
        # Test scaling with string
        m, n = random.randint(1, 3), random.randint(1, 3)
        A = 100 * np.random.rand(m, n)
        c = random.random()  # float in [0,1)
        A_ = MPMatrix.import_array(A)
        cA_np = c * A
        cA_mp = A_.scale(str(c))
        equality, log = _assert_mp_equals_np(cA_mp, cA_np)
        self.assertTrue(equality, msg=log)


class ZerosMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(1, 3), random.randint(1, 3)
        A_mp = MPMatrix.zeros(m, n)
        A_np = np.zeros((m, n))
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class get_rowMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(1, 3), random.randint(1, 3)
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = random.randint(0, m - 1)
        A_mp = A_.get_rows(k, 1)
        A_np = A[k].reshape((1, n))
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        return


class get_rowsMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(5, 10), random.randint(5, 10)
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = random.randint(0, m - 2)
        row_count = random.randint(2, m - k)
        A_mp = A_.get_rows(k, row_count)
        A_np = A[k:k + row_count]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        return


class get_all_rowsMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(5, 10), random.randint(5, 10)
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = random.randint(0, m - 2)
        A_mp = A_.get_rows(k, -1)
        A_np = A[k:]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        return


class drop_rowMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(1, 3), random.randint(1, 3)
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = random.randint(0, m - 1)
        A_mp = A_.drop_row(k)
        A_np = np.delete(A, k, 0)  # delete kth entry of 0th axis
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        return


class get_colMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(1, 3), random.randint(1, 3)
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = random.randint(0, n - 1)
        A_mp = A_.get_cols(k, 1)
        A_np = A[:, k].reshape((m, 1))
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        return


class get_colsMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(5, 10), random.randint(5, 10)
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = random.randint(0, n - 2)
        col_count = random.randint(2, n - k)
        A_mp = A_.get_cols(k, col_count)
        A_np = A[:, k:k + col_count]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        return


class get_all_colsMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(5, 10), random.randint(5, 10)
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = random.randint(0, n - 2)
        A_mp = A_.get_cols(k, -1)
        A_np = A[:, k:].reshape((m, n - k))
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        return


class MpfrEps(unittest.TestCase):
    def test(self):
        P = gmpy2.get_context().precision
        # This is the analyst's definition of machine epsilon
        # As in LAPACK and numerical papers, b^(-(P-1))/2,
        # where P includes the implicit bit.
        # The C, Python, Matlab standards define eps as half of this.
        epsilon = mpfr("0b0." + "0" * (P - 1) + "1")
        epsilon2 = mpfr(2**((-1) * P))
        self.assertTrue(epsilon.digits() == epsilon2.digits(),
                        msg="String comp: {}, float: {}".format(
                            epsilon.digits(), epsilon2.digits()))


if __name__ == '__main__':
    print("Testing at P=53.")
    unittest.main(exit=False)
    print("\nTesting at P=30.")
    gmpy2.get_context().precision = 30
    unittest.main(exit=False)
    print("\nTesting at P=80.")
    gmpy2.get_context().precision = 80
    unittest.main(exit=False)

    # TODO: change relative error so these work
    # print("\nTesting at P=4.")
    # gmpy2.get_context().precision = 4
    # unittest.main(exit=False)

    # print("\nTesting at P=2.")
    # gmpy2.get_context().precision = 2
    # unittest.main(exit=False)

    # print("\nTesting at P=1.")
    # gmpy2.get_context().precision = 1
    # unittest.main(exit=False)
