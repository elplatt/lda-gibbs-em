'''
Latent Dirichlet Allocation
'''

import math

import numpy as np
import scipy as sp
import scipy.misc as spmisc

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
        