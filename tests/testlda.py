import mock
import os
import sys
import unittest

import numpy as np
import numpy.testing as nptest

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

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
# Hyperparameters used in hand-calculations
stub_alpha = np.array([0.1, 0.2, 0.3])
stub_eta = np.array(range(1, 4)) / 100.0

# Hand-calculated conditional for m=0, w=7
topic_cond = np.array([2.08*4.1/9.05, 1.08*5.2/10.05, 2.08*3.3/8.05])
topic_cond /= topic_cond.sum()

# Hand-calculated per-topic word distributions
beta = np.array([
    np.array([1.01, 0.02, 1.03, 0.04, 0.05, 0.06, 1.07,
              2.08, 0.09, 0.10, 1.11, 0.12, 1.13, 1.14]) / 9.05,
    np.array([0.01, 1.02, 1.03, 0.04, 1.05, 1.06, 0.07,
              1.08, 1.09, 0.10, 0.11, 1.12, 1.13, 1.14]) / 10.05,
    np.array([0.01, 0.02, 1.03, 1.04, 0.05, 1.06, 0.07,
              2.08, 0.09, 2.10, 0.11, 0.12, 0.13, 0.14]) / 8.05
])

# Stub for numpy.randint over [0,3)
def stub_randint(min, max):
    stub_randint.count += 1
    val = [1,2,1,1,0,0,0,2,2,1,1,0, 1,2,1,0, 0,1,1, 0,2,2,2,0]
    return val[stub_randint.count]
stub_randint.count = -1

# Stub for numpy.random_sample
def stub_random_sample():
    stub_random_sample.count += 1
    val = [0.1, 0.3, 0.5, 0.7, 0.9]
    return val[stub_random_sample.count]
stub_random_sample.count = -1

class UtilTest(unittest.TestCase):
    
    def test_word_iter(self):
        '''Test iterating the words in the first document of the corpus.'''
        for m, correct_words in enumerate(corpus_words):
            words = list(word_iter(corpus[m,:]))
            self.assertEqual(words, correct_words)
            
    def test_sample(self):
        '''Test sampling from a discrete distribution'''
        tmp = __main__.nprand.random_sample
        __main__.nprand.random_sample = stub_random_sample
        try:
            dist = np.array([0.5, 0.25, 0.125, 0.125])
            samples = [0, 0, 1, 1, 3]
            test_samples = [sample(dist) for i in range(5)]
            nptest.assert_array_equal(test_samples, samples)
        finally:
            __main__.nprand.random_sample = tmp

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

class LdaBetaThetaTest(unittest.TestCase):
    
    def setUp(self):
        num_topics = 3
        vocab_size = 14
        alpha = np.array([0.1, 0.2, 0.3])
        eta = np.array(range(1, vocab_size+1)) / 100.0
        self.lda = LdaModel(corpus, num_topics, alpha, eta)
        self.lda.stats = stats
    
    def test_beta(self):
        test_beta = self.lda.beta()
        nptest.assert_allclose(test_beta, beta)

class LdaGibbsTest(unittest.TestCase):
    
    def setUp(self):
        num_topics = 3
        vocab_size = 14
        alpha = np.array([0.1, 0.2, 0.3])
        eta = np.array(range(1, vocab_size+1)) / 100.0
        self.lda = LdaModel(corpus, num_topics, alpha, eta)

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
    
    def test_gibbs_sample(self):
        # Run sampler to test for runtime errors
        stats = self.lda._gibbs_init(corpus)
        self.lda._gibbs_sample(stats)
        
if __name__ == '__main__':
    unittest.main()
