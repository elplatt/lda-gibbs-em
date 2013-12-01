'''
Latent Dirichlet Allocation

References:
Heinrich, Gregor. Parameter estimation for text analysis. Technical report. 2005 (revised 2009)
Minka, Thomas P. Estimating a Dirichlet distribution. 2000. (revised 2012)
'''

import math

import numpy as np
import numpy.random as nprand
import scipy as sp
import scipy.misc as spmisc
import scipy.special as spspecial

def word_iter(doc):
    '''Return an iterator over words in a document.
    :param doc: doc[w] is the number of times word w appears
    '''
    for w, count in enumerate(doc):
        for i in xrange(count):
            yield w

def sample(dist):
    '''Sample from the given distribution.
    
    :param dist: array of probabilities
    :returns: a randomly sampled integer in [0, len(dist))
    '''
    cdf = np.cumsum(dist)
    uniform = nprand.random_sample()
    return next(n for n in range(0, len(cdf)) if cdf[n] > uniform)

def polya_iteration(ndm, nd, guess, iter=5):
    '''Estimate the parameter of a dirichlet-multinomial distribution
    with num_dir draws from the dirichlet and num_out possible outcomes.
    
    :param ndm: Counts as (num_dir, num_out) array
    :param nd: Counts as (num_dir) array
    :param guess: Initial guess for dirichlet parameter
    :param iter: Number of iterations to perform, default 5
    :returns: An updated estimate of the dirichlet parameter
    '''
    num_dir, num_out = ndm.shape
    param = guess
    for i in range(iter):
        # Heinrich2005 Eq. 83
        # Minka2000 Eq. 55
        new = np.zeros(num_out)
        param_sum = param.sum()
        den = 0
        for d in range(num_dir):
            den += spspecial.psi(nd[d] + param_sum)
        den -= num_dir * spspecial.psi(param_sum)
        for m in range(num_out):
            for d in range(num_dir):
                new[m] += spspecial.psi(ndm[d,m] + param[m])
            new[m] -= num_dir * spspecial.psi(param[m])
            new[m] /= den
            new[m] *= param[m]
            if new[m] <= 0:
                new[m] = 1e-5
        param = new
    return param

class LdaModel(object):
    
    def __init__(self, training, num_topics, alpha=0.1, eta=0.1):
        '''Creates an LDA model.
        
        :param training: training corpus as (num_doc, vocab_size) array
        :param num_topics: number of topics
        :param alpha: document-topic dirichlet parameter, scalar or array,
            defaults to 0.1
        :param eta: topic-word dirichlet parameter, scalar or array,
            defaults to 0.1
        '''
        self.num_topics = num_topics
        # Validate alpha and eta, and convert to array if necessary
        try:
            if len(alpha) != num_topics:
                raise ValueError("alpha must be a number or a num_topic-length vector")
            self.alpha = alpha
        except TypeError:
            self.alpha = np.ones(num_topics)*alpha
        try:
            if len(eta) != training.shape[1]:
                raise ValueError("eta must be a number or a vocab_size-length vector")
            self.eta = eta
        except TypeError:
            self.eta = np.ones(training.shape[1])*eta
        # Initialize gibbs sampler
        self.stats = self._gibbs_init(training)
    
    def e_step(self, gibbs_iter=50):
        '''Associate each word with a topic using Gibbs sampling.'''
        for i in range(gibbs_iter):
            self._gibbs_sample(self.stats)
            
    def m_step(self):
        '''Update estimates for alpha and eta to maximize likelihood.'''
        self._m_alpha()
        self._m_eta()
    
    def beta(self):
        '''Per-topic word distributions as a (num_topics, vocab_size) array.'''
        result = self.stats['nkw'] + self.eta
        result = np.divide(result, result.sum(1)[:,np.newaxis])
        return result
    
    def _gibbs_init(self, corpus):
        '''Initialize Gibbs sampling by assigning a random topic to each word in
            the corpus.
        :param corpus: corpus[m][w] is the count for word w in document m
        :returns: statistics dict with the following keys:
            nmk: document-topic count, nmk[m][k] is for document m, topic k
            nm: document-topic sum, nm[m] is the number of words in document m
            nkw: topic-term count, nkw[k][w] is for word w in topic k
            nk: topic-term sum, nk[k] is the count of topic k in corpus
            n: total number of words
            topics: list of pairs (m, i) giving the topic for each word
        '''
        num_docs, num_words = corpus.shape
        # Initialize stats
        stats = {
            'nmk': np.zeros((num_docs, self.num_topics))
            , 'nm': np.zeros(num_docs)
            , 'nkw': np.zeros((self.num_topics, num_words))
            , 'nk': np.zeros(self.num_topics)
            , 'n': 0
            , 'topics': {}
        }
        for m in xrange(num_docs):
            for i, w in enumerate(word_iter(corpus[m,:])):
                # Sample topic from uniform distribution
                k = nprand.randint(0, self.num_topics)
                stats['nmk'][m][k] += 1
                stats['nm'][m] += 1
                stats['nkw'][k][w] += 1
                stats['nk'][k] += 1
                stats['n'] += 1
                stats['topics'][(m, i)] = (w, k)
        return stats
    
    def _gibbs_sample(self, stats):
        '''Resample topics for each word using Gibbs sampling.
        
        :param stats: statistics returned by _gibbs_init(), will be modified.
        '''
        # Shuffle topic assignments
        topics = stats['topics'].keys()
        nprand.shuffle(topics)
        # Resample each one
        for m, i in topics:
            # Remove doc m, word i from stats
            w, k = stats['topics'][(m, i)]
            stats['nmk'][m][k] -= 1
            stats['nm'][m] -= 1
            stats['nkw'][k][w] -= 1
            stats['nk'][k] -= 1
            # Sample from conditional
            k = sample(self.topic_conditional(m, w, stats))
            # Add new topic to stats
            stats['nmk'][m][k] += 1
            stats['nm'][m] += 1
            stats['nkw'][k][w] += 1
            stats['nk'][k] += 1
            stats['topics'][(m, i)] = (w, k)
    
    def _m_alpha(self, iter=5):
        '''Find a new estimate for alpha that maximizes likelihood.
        
        :param iter: The number of iterations to perform, defaults to 5
        '''
        ndm = self.stats['nmk']
        nd = self.stats['nm']
        self.alpha = polya_iteration(ndm, nd, self.alpha, iter)
        
    def _m_eta(self, iter=5):
        '''Find a new estimate for alpha that maximizes likelihood.
        
        :param iter: The number of iterations to perform, defaults to 5
        '''
        ndm = self.stats['nkw']
        nd = self.stats['nk']
        self.eta = polya_iteration(ndm, nd, self.eta, iter)
    
    def topic_conditional(self, m, w, stats):
        '''Distribution of a single topic given others and words.
        
        :param m: index of the document to sample for
        :param w: word associated with the topic being sampled
        :param stats: count statistics (with topic being sampled removed)
        :returns: a (num_topics) length vector of topic probabilities
        '''
        pk = stats['nkw'][:,w] + self.eta[w]
        pk = np.multiply(pk, stats['nmk'][m,:] + self.alpha)
        pk = np.divide(pk, stats['nk'] + self.eta.sum())
        # Normalize
        pk /= pk.sum()
        return pk