import unittest
import numpy as np
import gmpy2
from gmpy2 import mpfr  # float class
import itertools
import random
from MPMatrix import MPMatrix
from guppy import hpy

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
    if epsilon == None:
        P = gmpy2.get_context().precision
        epsilon = mpfr("0b" + "0" * P + "1")
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


class GetRowViewFromRect(unittest.TestCase):
    def test(self):
        m, n = 4, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = 3
        A_mp = A_[k, :]
        A_np = A[np.newaxis, k, :]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetColViewFromRect(unittest.TestCase):
    def test(self):
        m, n = 4, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = 3
        A_mp = A_[:, k]
        A_np = A[:, k, np.newaxis]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality,
                        msg=log + "\nMP: {}, \n\nNP: {}".format(A_mp, A_np))


class GetPtViewFromRect(unittest.TestCase):
    def test(self):
        m, n = 4, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_mp = A_[2:3, 2:3]
        A_np = A[2:3, 2:3]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetRowViewFromCol(unittest.TestCase):
    def test(self):
        m, n = 4, 1
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = 3
        A_mp = A_[k, :]
        A_np = A[np.newaxis, k, :]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetColViewFromCol(unittest.TestCase):
    def test(self):
        m, n = 4, 1
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        k = 0
        A_mp = A_[:, k]
        A_np = A[k, :, np.newaxis]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetPtViewFromCol(unittest.TestCase):
    def test(self):
        m, n = 4, 1
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_mp = A_[2:3, :]
        A_np = A[2:3, :]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetPtFromCol(unittest.TestCase):
    """Test single-int indexing"""
    def test(self):
        m, n = 4, 1
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_mp = A_[2]
        A_np = A[2, 0]
        equality = _ptwise_vals_equal(A_mp, A_np, None)
        self.assertTrue(equality, msg="MP: {}, NP: {}".format(A_mp, A_np))


class GetRowViewFromRow(unittest.TestCase):
    def test(self):
        m, n = 1, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_mp = A_[0, 1:]
        A_np = A[:, 1:]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetColViewFromRow(unittest.TestCase):
    def test(self):
        m, n = 1, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_mp = A_[:, 0]
        A_np = A[:, 0:1]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetPtViewFromRow(unittest.TestCase):
    def test(self):
        m, n = 1, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_mp = A_[:, 2:3]
        A_np = A[:, 2:3]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetPtFromRow(unittest.TestCase):
    """Test single-int indexing"""
    def test(self):
        m, n = 1, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_mp = A_[2]
        A_np = A[0, 2]
        equality = _ptwise_vals_equal(A_mp, A_np, None)
        self.assertTrue(equality, msg="MP: {}, NP: {}".format(A_mp, A_np))


class GetPtViewFromPtView(unittest.TestCase):
    def test(self):
        m, n = 1, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_np = A[:, 2:3]
        A_int = A_[[0], [2]]
        A_mp = A_int[[0], [0]]
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class GetPtFromPtView(unittest.TestCase):
    def test(self):
        m, n = 1, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        A_np = A[0, 2]
        A_int = A_[[0], [2]]
        A_mp = A_int[0]
        equality = _ptwise_vals_equal(A_mp, A_np, None)
        self.assertTrue(equality, msg="MP: {}, NP: {}".format(A_mp, A_np))
        self.assertTrue(A_mp == A_int[0, 0],
                        msg=": {}, NP: {}".format(A_mp, A_int[0, 0]))


class SetRow(unittest.TestCase):
    def test(self):
        m, n = 6, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        B = 100 * np.random.rand(m, n)
        B_ = MPMatrix.import_array(B)
        A_[2, :] = B_[2, :]
        A[2, :] = B[2, :]
        A_mp = A_
        A_np = A
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


class SetCol(unittest.TestCase):
    def test(self):
        m, n = 6, 5
        A = 100 * np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        B = 100 * np.random.rand(m, n)
        B_ = MPMatrix.import_array(B)
        A_[:, 2] = B_[:, 2]
        A[:, 2] = B[:, 2]
        A_mp = A_
        A_np = A
        equality, log = _assert_mp_equals_np(A_mp, A_np)
        self.assertTrue(equality, msg=log)


def multiprec_test():
    print("Testing at P=53.")
    unittest.main(exit=False)
    print("\nTesting at P=30.")
    gmpy2.get_context().precision = 30
    unittest.main(exit=False)
    print("\nTesting at P=80.")
    gmpy2.get_context().precision = 80
    unittest.main(exit=False)


def singleton_test(test):
    """run single test: arg of the form TestName(methodName='test'))"""
    suite = unittest.TestSuite()
    suite.addTest(test)
    runner = unittest.TextTestRunner()
    runner.run(suite)


def test_views():
    suite = unittest.TestSuite()
    suite.addTest(GetRowViewFromRect(methodName='test'))
    suite.addTest(GetColViewFromRect(methodName='test'))
    suite.addTest(GetPtViewFromRect(methodName='test'))
    suite.addTest(GetRowViewFromCol(methodName='test'))
    suite.addTest(GetColViewFromCol(methodName='test'))
    suite.addTest(GetPtViewFromCol(methodName='test'))
    suite.addTest(GetPtFromCol(methodName='test'))
    suite.addTest(GetRowViewFromRow(methodName='test'))
    suite.addTest(GetColViewFromRow(methodName='test'))
    suite.addTest(GetPtViewFromRow(methodName='test'))
    suite.addTest(GetPtFromRow(methodName='test'))
    suite.addTest(GetPtViewFromPtView(methodName='test'))
    suite.addTest(GetPtFromPtView(methodName='test'))

    runner = unittest.TextTestRunner()
    runner.run(suite)


def memory_check():
    m, n = 1000, 1000
    A = 100 * np.random.rand(m, n)
    A_ = MPMatrix.import_array(A)
    print(hpy().heap())
    B = A_[1:, :]
    print("Post-view:")
    print(hpy().heap())


if __name__ == '__main__':
    memory_check()
    #test_views()
    #multiprec_test()

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
