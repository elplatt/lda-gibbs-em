import mock
import os
import sys
import unittest

import numpy as np
import numpy.testing as nptest

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import __main__
import lda

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
    , 'topics': dict()
}
# Fake data for testing stat merging/splitting
query_corpus = np.array([
    [1,0,0,1,0,0,0,1,0,1,1,0,0,1],
    [0,0,1,0,0,0,0,2,0,1,0,0,1,1]
])
query_stats = {
    'nmk': np.array([[2, 1, 3], [2, 3, 1]])
    , 'nm': np.array([6, 6])
    , 'nkw': np.array([
        [0,0,0,1,0,0,0,0,0,1,0,0,0,2],
        [0,0,1,0,0,0,0,2,0,0,0,0,1,0],
        [1,0,0,0,0,0,0,1,0,1,1,0,0,0]
    ])
    , 'nk': np.array([4, 4, 4])
    , 'topics': {
        (0,0):(13,0),(0,1):(10,2),(0,2):(3,0),(0,3):(7,1),(0,4):(9,2),(0,5):(0,2)
        , (1,0):(12,1),(1,1):(7,2),(1,2):(9,0),(1,3):(13,0),(1,4):(7,1),(1,5):(2,1)
    }
}
merged_stats = {
    'nmk': np.array([[2,1,3],[2,3,1],[4,5,3],[1,2,1],[1,2,0],[2,0,3]])
    , 'nm': np.array([6, 6, 12, 4, 3, 5])
    , 'nkw': np.array([
        [1,0,1,1,0,0,1,2,0,1,1,0,1,3],
        [0,1,2,0,1,1,0,3,1,0,0,1,2,1],
        [1,0,1,1,0,1,0,3,0,3,1,0,0,0]
    ])
    , 'nk': np.array([12, 13, 11])
    , 'topics': {
        (0,0):(13,0),(0,1):(10,2),(0,2):(3,0),(0,3):(7,1),(0,4):(9,2),(0,5):(0,2)
        , (1,0):(12,1),(1,1):(7,2),(1,2):(9,0),(1,3):(13,0),(1,4):(7,1),(1,5):(2,1)
    }
}
# Hyperparameters used in hand-calculations
stub_alpha = np.array([0.1, 0.2, 0.3])
stub_eta = np.array(range(1, 15)) / 100.0

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

# Hand-calculated per-document topic distribution
theta = np.array([
    np.array([4.1, 5.2, 3.3]) / 12.6
    , np.array([1.1, 2.2, 1.3]) / 4.6
    , np.array([1.1, 2.2, 0.3]) / 3.6
    , np.array([2.1, 0.2, 3.3]) / 5.6
])

# Hand-calculated query likelihood and perplexity
query_log_lik = np.array([-15.2329, -13.1347])
query_perp = 10.633

# Stub for sampling query topics
def stub_sample_query(dist):
    stub_sample_query.count += 1
    val = [2, 0, 1, 2, 2, 0, 1, 2, 1, 0, 1, 0]
    return val[stub_sample_query.count]
stub_sample_query.count = -1

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

# Stub for (in place) numpy.random.shuffle
def stub_shuffle(l):
    l.sort()
    
# Stub for query
def stub_query(corpus):
    return query_stats

class UtilTest(unittest.TestCase):
    
    def test_word_iter(self):
        '''Test iterating the words in the first document of the corpus.'''
        for m, correct_words in enumerate(corpus_words):
            words = list(lda.word_iter(corpus[m,:]))
            self.assertEqual(words, correct_words)
            
    def test_sample(self):
        '''Test sampling from a discrete distribution'''
        tmp = lda.nprand.random_sample
        lda.nprand.random_sample = stub_random_sample
        try:
            dist = np.array([0.5, 0.25, 0.125, 0.125])
            samples = [0, 0, 1, 1, 3]
            test_samples = [lda.sample(dist) for i in range(5)]
            nptest.assert_array_equal(test_samples, samples)
        finally:
            lda.nprand.random_sample = tmp
        
    def test_merge_query_stats(self):
        test_stats = lda.merge_query_stats(stats, query_stats)
        nptest.assert_array_equal(test_stats['nmk'], merged_stats['nmk'])
        nptest.assert_array_equal(test_stats['nm'], merged_stats['nm'])
        nptest.assert_array_equal(test_stats['nkw'], merged_stats['nkw'])
        nptest.assert_array_equal(test_stats['nk'], merged_stats['nk'])
        nptest.assert_array_equal(test_stats['nk'], merged_stats['nk'])
        test_topics = sorted(test_stats['topics'].items())
        merged_topics = sorted(merged_stats['topics'].items())
        nptest.assert_array_equal(test_topics, merged_topics)
        
    def test_split_query_stats(self):
        test_stats = lda.split_query_stats(stats, merged_stats)
        nptest.assert_array_equal(test_stats['nmk'], query_stats['nmk'])
        nptest.assert_array_equal(test_stats['nm'], query_stats['nm'])
        nptest.assert_array_equal(test_stats['nkw'], query_stats['nkw'])
        nptest.assert_array_equal(test_stats['nk'], query_stats['nk'])
        nptest.assert_array_equal(test_stats['nk'], query_stats['nk'])
        test_topics = sorted(test_stats['topics'].items())
        query_topics = sorted(query_stats['topics'].items())
        nptest.assert_array_equal(test_topics, query_topics)
    
    def test_multinomial_beta(self):
        test = lda.log_multinomial_beta(stub_alpha + stats['nmk'], 1)
        lmb = np.array([-12.5938, -2.6550, -0.1701, -1.5633])
        nptest.assert_allclose(test, lmb, rtol=1e-3)

class LdaInitTest(unittest.TestCase):
    
    def setUp(self):
        self.num_topics = 10
        self.vocab_size = 14
    
    def test_init_scalar(self):
        alpha = 0.2
        eta = 0.3
        model = lda.LdaModel(corpus, self.num_topics, alpha, eta)
        nptest.assert_array_equal(model.alpha, np.ones(self.num_topics)*alpha)
        nptest.assert_array_equal(model.eta, np.ones(self.vocab_size)*eta)
    
    def test_init_vector(self):
        alpha = np.ones(self.num_topics) * 0.2
        eta = np.ones(self.vocab_size) * 0.3
        model = lda.LdaModel(corpus, self.num_topics, alpha, eta)
        nptest.assert_array_equal(model.alpha, alpha)
        nptest.assert_array_equal(model.eta, eta)
    
    def test_init_default(self):
        model = lda.LdaModel(corpus, self.num_topics)
        alpha = np.ones(self.num_topics) * 0.1
        eta = np.ones(self.vocab_size) * 0.1
        nptest.assert_array_equal(model.alpha, alpha)
        nptest.assert_array_equal(model.eta, eta)

class LdaBetaThetaTest(unittest.TestCase):
    
    def setUp(self):
        num_topics = 3
        vocab_size = 14
        alpha = np.array([0.1, 0.2, 0.3])
        eta = np.array(range(1, vocab_size+1)) / 100.0
        self.model = lda.LdaModel(corpus, num_topics, alpha, eta)
        self.model.stats = stats
    
    def test_beta(self):
        test_beta = self.model.beta()
        nptest.assert_allclose(test_beta, beta)
        
    def test_theta(self):
        test_theta = self.model.theta()
        nptest.assert_allclose(test_theta, theta)

class LdaGibbsTest(unittest.TestCase):
    
    def setUp(self):
        num_topics = 3
        vocab_size = 14
        alpha = np.array([0.1, 0.2, 0.3])
        eta = np.array(range(1, vocab_size+1)) / 100.0
        self.model = lda.LdaModel(corpus, num_topics, alpha, eta, 0, 0)

    def test_gibbs_init(self):
        # Use random stub
        tmp = lda.nprand.randint
        lda.nprand.randint = stub_randint
        try:
            test_stats = self.model._gibbs_init(corpus)
            nptest.assert_array_equal(test_stats['nmk'], stats['nmk'])
            nptest.assert_array_equal(test_stats['nm'], stats['nm'])
            nptest.assert_array_equal(test_stats['nkw'], stats['nkw'])
            nptest.assert_array_equal(test_stats['nk'], stats['nk'])
        finally:
            lda.nprand.randint = tmp
            
    def test_topic_conditional(self):
        test_cond = self.model.topic_conditional(0, 7, stats)
        nptest.assert_allclose(test_cond, topic_cond)
    
    def test_topic_conditional_norm(self):
        test_cond = self.model.topic_conditional(0, 7, stats)
        self.assertAlmostEqual(test_cond.sum(), 1.0)
    
    def test_gibbs_sample(self):
        # Run sampler to test for runtime errors
        stats = self.model._gibbs_init(corpus)
        self.model._gibbs_sample(stats)

class LdaQueryTest(unittest.TestCase):
    
    def setUp(self):
        num_topics = 3
        vocab_size = 14
        alpha = np.array([0.1, 0.2, 0.3])
        eta = np.array(range(1, vocab_size+1)) / 100.0
        self.model = lda.LdaModel(corpus, num_topics, alpha, eta, 0, 0)
        self.model.stats = stats
    
    def test_log_lik(self):
        lik = self.model.log_likelihood(query_corpus, query_stats)
        nptest.assert_allclose(lik, query_log_lik, rtol=1e-2)
    
    def test_perplexity(self):
        old_query = self.model.query
        self.model.query = stub_query
        try:
            perplexity = self.model.perplexity(query_corpus)
            self.assertAlmostEqual(perplexity, query_perp, places=3)
        finally:
            self.model.query = old_query
    
    def test_query(self):
        old_shuffle = lda.nprand.shuffle
        lda.nprand.shuffle = stub_shuffle
        old_sample = lda.sample
        lda.sample = stub_sample_query
        try:
            test_stats = self.model.query(query_corpus)
            nptest.assert_array_equal(test_stats['nmk'], query_stats['nmk'])
            nptest.assert_array_equal(test_stats['nm'], query_stats['nm'])
            nptest.assert_array_equal(test_stats['nkw'], query_stats['nkw'])
            nptest.assert_array_equal(test_stats['nk'], query_stats['nk'])
        finally:
            lda.nprand.shuffle = old_shuffle
            lda.sample = old_sample

class LdaEmTest(unittest.TestCase):
    
    def setUp(self):
        num_topics = 3
        vocab_size = 14
        self.model = lda.LdaModel(corpus, num_topics, stub_alpha, stub_eta)
        self.model.stats = stats
    
    def test_m_alpha(self):
        new_alpha = np.array([0.1*42.63, 0.2*18.5, 0.3*12.42]) / 12.83
        self.model._m_alpha(1)
        nptest.assert_allclose(self.model.alpha, new_alpha, rtol=1e-2)

if __name__ == '__main__':
    unittest.main()
