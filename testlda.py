import unittest

import numpy as np
import numpy.testing as nptest

from lda import LdaModel

class LdaInitTest(unittest.TestCase):
    
    def SetUp(self):
        pass
    
    def test_init_scalar(self):
        num_topics = 10
        alpha = 0.2
        eta = 0.3
        lda = LdaModel(num_topics, alpha, eta)
        nptest.assert_array_equal(lda.alpha, np.ones(num_topics)*alpha)
        nptest.assert_array_equal(lda.eta, np.ones(num_topics)*eta)
    
    def test_init_vector(self):
        num_topics = 10
        alpha = np.ones(num_topics) * 0.2
        eta = np.ones(num_topics) * 0.3
        lda = LdaModel(num_topics, alpha, eta)
        nptest.assert_array_equal(lda.alpha, alpha)
        nptest.assert_array_equal(lda.eta, eta)
    
    def test_init_default(self):
        num_topics = 10
        lda = LdaModel(num_topics)
        alpha = np.ones(num_topics) * 0.1
        eta = np.ones(num_topics) * 0.1
        nptest.assert_array_equal(lda.alpha, alpha)
        nptest.assert_array_equal(lda.eta, eta)
    
if __name__ == '__main__':
    unittest.main()
