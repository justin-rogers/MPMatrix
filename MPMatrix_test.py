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
    ptwise_diff = abs(mp_val - mpfr(np_val))
    return ptwise_diff < epsilon


def _assert_arrays_equal(mp_array, np_array):
    """Compares an MPMatrix and np array, checking for equality"""
    coordinates = itertools.product(range(m), range(n))
    P = gmpy2.get_context().precision
    epsilon = mpfr("0b" + "0" * P + "1")
    for coord in coordinates:
        mp_val = mp_array[coord]
        np_val = np_array[coord]
        self.assertTrue(
            _ptwise_vals_equal(mp_val, np_val, epsilon),
            msg="Distinct values mp: {}, np: {} at coord {}".format(
                mp_val, np_val, coord))


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
        A = np.random.rand(n, n)
        A_ = MPMatrix.import_array(A)
        coordinates = itertools.product(range(n), range(n))
        P = gmpy2.get_context().precision
        epsilon = mpfr("0b" + "0" * P + "1")
        for coord in coordinates:
            mp_val = A_[coord]
            np_val = A[coord]
            ptwise_diff = abs(mp_val - mpfr(np_val))
            self.assertTrue(ptwise_diff < epsilon)


class AddMP(unittest.TestCase):
    def test(self):
        m, n = random.randint(1, 3), random.randint(1, 3)
        A, B = np.random.rand(m, n), np.random.rand(m, n)
        A_ = MPMatrix.import_array(A)
        B_ = MPMatrix.import_array(B)
        coordinates = itertools.product(range(m), range(n))
        P = gmpy2.get_context().precision
        epsilon = mpfr("0b" + "0" * P + "1")
        for coord in coordinates:
            mp_val = (A_ + B_)[coord]
            np_val = (A + B)[coord]
            ptwise_diff = abs(mp_val - mpfr(np_val))
            self.assertTrue(ptwise_diff < epsilon)


class MatMulMP(unittest.TestCase):
    def test(self):
        # Test multiplying (m,k)*(k,n) matrices
        m, n = random.randint(1, 3), random.randint(1, 3)
        k = random.randint(1, 3)
        A, B = np.random.rand(m, k), np.random.rand(k, n)
        A_ = MPMatrix.import_array(A)
        B_ = MPMatrix.import_array(B)
        AB_np = np.matmul(A, B)
        AB_mp = A_ * B_
        coordinates = itertools.product(range(m), range(n))
        P = gmpy2.get_context().precision
        epsilon = mpfr("0b" + "0" * P + "1")
        for coord in coordinates:
            mp_val = AB_mp[coord]
            np_val = AB_np[coord]
            ptwise_diff = abs(mp_val - mpfr(np_val))

            log = ('\nn: {}, m: {}, k: {}\n'.format(n, m, k) +
                   'i: {}, j: {}\n'.format(coord[0], coord[1]) +
                   'MPMatrix value {}, array value {}, cast to {}\n'.format(
                       mp_val, np_val, mpfr(np_val)) +
                   'Diff: {}\n Epsilon: {}\n'.format(ptwise_diff, epsilon))
            self.assertTrue(ptwise_diff < epsilon, msg=log)


class ScaleFloatMP(unittest.TestCase):
    def test(self):
        # Test scaling
        m, n = random.randint(1, 3), random.randint(1, 3)
        A = np.random.rand(m, n)
        c = random.random()  # float in [0,1)
        A_ = MPMatrix.import_array(A)
        cA_np = c * A
        cA_mp = A_.scale(c)
        coordinates = itertools.product(range(m), range(n))
        P = gmpy2.get_context().precision
        epsilon = mpfr("0b" + "0" * P + "1")
        for coord in coordinates:
            mp_val = cA_mp[coord]
            np_val = cA_np[coord]
            ptwise_diff = abs(mp_val - mpfr(np_val))
            self.assertTrue(ptwise_diff < epsilon,
                            msg="\nA: {}\nc: {}".format(A, c))


class ScaleStringMP(unittest.TestCase):
    def test(self):
        # Test scaling with string
        m, n = random.randint(1, 3), random.randint(1, 3)
        A = np.random.rand(m, n)
        c = random.random()  # float in [0,1)
        A_ = MPMatrix.import_array(A)
        cA_np = c * A
        cA_mp = A_.scale(str(c))
        coordinates = itertools.product(range(m), range(n))
        P = gmpy2.get_context().precision
        epsilon = mpfr("0b" + "0" * P + "1")
        for coord in coordinates:
            mp_val = cA_mp[coord]
            np_val = cA_np[coord]
            ptwise_diff = abs(mp_val - mpfr(np_val))
            self.assertTrue(ptwise_diff < epsilon,
                            msg="\nA: {}\nc: {}".format(A, c))


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
    print("\nTesting at P=33.")
    gmpy2.get_context().precision -= 20
    unittest.main(exit=False)
    print("\nTesting at P=83.")
    gmpy2.get_context().precision += 50
    unittest.main()
