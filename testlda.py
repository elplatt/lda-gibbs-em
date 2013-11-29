import mock
import sys
import unittest

import numpy as np
import numpy.testing as nptest

import __main__
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
corpus_words = [
    [1,2,4,5,6,7,7,7,9,11,12,13]
    , [2,3,7,10]
    , [2,8,13]
    , [0,5,7,9,12]
]
# Hand-calculated stats based on random stub
stats = {
    'nmk': np.array([[4,5,3], [1,2,1], [1,2,0], [2, 0, 3]])
    ,'nm': np.array([12, 4, 3, 5])
    ,'nkw': np.array([
        [1,0,1,0,0,0,1,2,0,0,1,0,1,1],
        [0,1,1,0,1,1,0,1,1,0,0,1,1,1],
        [0,0,1,1,0,1,0,2,0,2,0,0,0,0]
    ])
    , 'nk': np.array([8, 9, 7])
}
# Hand-calculated conditional for m=0, w=7
topic_cond = np.array([2.1*4.1/9.05, 1.2*5.2/10.05, 2.3*3.3/8.05])

# Stub for numpy.randint over [0,3)
def stub_randint(min, max):
    stub_randint.count += 1
    val = [1,2,1,1,0,0,0,2,2,1,1,0, 1,2,1,0, 0,1,1, 0,2,2,2,0]
    return val[stub_randint.count]
stub_randint.count = -1

class UtilTest(unittest.TestCase):
    
    def test_word_iter(self):
        '''Test iterating the words in the first document of the corpus.'''
        for m, correct_words in enumerate(corpus_words):
            words = list(word_iter(corpus[m,:]))
            self.assertEqual(words, correct_words)

class LdaInitTest(unittest.TestCase):
    
    def setUp(self):
        self.num_topics = 10
        self.vocab_size = 14
    
    def test_init_scalar(self):
        alpha = 0.2
        eta = 0.3
        lda = LdaModel(corpus, self.num_topics, alpha, eta)
        nptest.assert_array_equal(lda.alpha, np.ones(self.num_topics)*alpha)
        nptest.assert_array_equal(lda.eta, np.ones(self.vocab_size)*eta)
    
    def test_init_vector(self):
        alpha = np.ones(self.num_topics) * 0.2
        eta = np.ones(self.vocab_size) * 0.3
        lda = LdaModel(corpus, self.num_topics, alpha, eta)
        nptest.assert_array_equal(lda.alpha, alpha)
        nptest.assert_array_equal(lda.eta, eta)
    
    def test_init_default(self):
        lda = LdaModel(corpus, self.num_topics)
        alpha = np.ones(self.num_topics) * 0.1
        eta = np.ones(self.vocab_size) * 0.1
        nptest.assert_array_equal(lda.alpha, alpha)
        nptest.assert_array_equal(lda.eta, eta)

class LdaGibbsTest(unittest.TestCase):
    
    def setUp(self):
        self.lda = LdaModel(corpus, 3)

    def test_gibbs_init(self):
        # Use random stub
        tmp = __main__.nprand.randint
        __main__.nprand.randint = stub_randint
        try:
            test_stats = self.lda._gibbs_init(corpus)
            nptest.assert_array_equal(test_stats['nmk'], stats['nmk'])
            nptest.assert_array_equal(test_stats['nm'], stats['nm'])
            nptest.assert_array_equal(test_stats['nkw'], stats['nkw'])
            nptest.assert_array_equal(test_stats['nk'], stats['nk'])
        finally:
            __main__.nprand.randint = tmp
            
    def test_topic_conditional(self):
        test_cond = self.lda.topic_conditional(0, 7, stats)
        nptest.assert_allclose(test_cond, topic_cond)
    
    def test_topic_conditional_norm(self):
        test_cond = self.lda.topic_conditional(0, 7, stats)
        self.assertAlmostEqual(test_cond.sum(), 1.0)
            
if __name__ == '__main__':
    unittest.main()
