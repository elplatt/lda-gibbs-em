import mock
import unittest

import numpy as np
import numpy.testing as nptest

from lda import *

# Sample corpus for validating calculations
text = [
    "who on first what on second i don't know is on third"
    , "the guy on first"
    , "who playing first"
    , "what is on second base"
]
vocab = {'on': 7, 'what': 12, 'third': 11, "don't": 1, 'i': 4, 'is': 5, 'who': 13, 'the': 10, 'second': 9, 'base': 0, 'know': 6, 'guy': 3, 'playing': 8, 'first': 2}
corpus = np.array([
    [0,1,1,0,1,1,1,3,0,1,0,1,1,1],
    [0,0,1,1,0,0,0,1,0,0,1,0,0,0],
    [0,0,1,0,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,1,0,0,1,0]
])

class UtilTest(unittest.TestCase):
    
    def test_word_iter(self):
        '''Test iterating the words in the first document of the corpus.'''
        words = [1,2,4,5,6,7,7,7,9,11,12,13]
        self.assertEqual(list(word_iter(corpus[0,:])), words)

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
