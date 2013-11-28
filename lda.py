'''
Latent Dirichlet Allocation
'''

import math

import numpy as np
import numpy.random as nprand
import scipy as sp
import scipy.misc as spmisc

def word_iter(doc):
    '''Return an iterator over words in a document.
    :param doc: doc[w] is the number of times word w appears
    '''
    for w, count in enumerate(doc):
        for i in xrange(count):
            yield w

class LdaModel(object):
    
    def __init__(self, num_topics, alpha=0.1, eta=0.1):
        '''Creates an LDA model.
        
        :param num_topics: number of topics
        :param alpha: document-topic dirichlet parameter, scalar or array,
            defaults to 0.1
        :param eta: topic-word dirichlet parameter, scalar or array,
            defaults to 0.1
        '''
        self.num_topics = num_topics
        # Validate alpha, eta and convert to array if necessary
        try:
            if len(alpha) != num_topics:
                raise ValueError("alpha must be a number or a num_topic-length vector")
            self.alpha = alpha
        except TypeError:
            self.alpha = np.ones(num_topics)*alpha
        try:
            if len(eta) != num_topics:
                raise ValueError("eta must be a number or a num_topic-length vector")
            self.eta = eta
        except TypeError:
            self.eta = np.ones(num_topics)*eta
        
    def _gibbs_init(self, corpus):
        '''Initialize Gibbs sampling by assigning a random topic to each word in
            the corpus.
        :param corpus: corpus[m][w] is the count for word w in document m
        :returns: statistics dict with the following keys:
            nmk: document-topic count, nmk[m][k] is for document m, topic k
            nm: document-topic sum, nm[m] is the number of words in document m
            nkw: topic-term count, nkw[k][w] is for word w in topic k
            nk: topic-term sum, nk[k] is the count of topic k in corpus
        '''
        num_docs, num_words = corpus.shape
        # Initialize stats
        stats = {
            'nmk': np.zeros((num_docs, self.num_topics))
            , 'nm': np.zeros(num_docs)
            , 'nkw': np.zeros((self.num_topics, num_words))
            , 'nk': np.zeros(self.num_topics)
        }
        for m in xrange(num_docs):
            for w in word_iter(corpus[m,:]):
                # Sample topic from uniform distribution
                k = nprand.randint(0, self.num_topics)
                stats['nmk'][m][k] += 1
                stats['nm'][m] += 1
                stats['nkw'][k][w] += 1
                stats['nk'][k] += 1
        return stats