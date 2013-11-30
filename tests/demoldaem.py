'''
Generate data from a known model.
Perform both E and M step to find alpha and eta that maximize the likelihood.
'''

import os
import sys

import numpy as np
import numpy.random as nprand
import scipy.misc as spmisc

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import lda

VOCAB_SIZE = 100
NUM_TOPICS = 4
NUM_DOCS = 100
DOC_LENGTH = 100
NUM_ITER = 10
NUM_GIBBS = 20

alpha = np.array([0.02, 0.02, 0.02, 0.02])
eta = 0.1*np.ones(VOCAB_SIZE)

def main():
    print 'Generating corpus'
    corpus = generate_corpus()
    alpha_guess = 0.5
    print 'Initializing model'
    model = lda.LdaModel(corpus, NUM_TOPICS, alpha_guess, eta)
    for i in range(NUM_ITER):
        print 'Iteration %d' % i
        lda.e_step()
        # M-step
        lda.m_step()
        print 'alpha %d:' % i
        print model.alpha

def generate_corpus():
    beta = np.zeros((NUM_TOPICS, VOCAB_SIZE))
    corpus = np.zeros((NUM_DOCS, VOCAB_SIZE), dtype='int64')
    # Draw per-topic word distributions
    for k in range(NUM_TOPICS):
        beta[k,:] = nprand.dirichlet(eta)
    for m in range(NUM_DOCS):
        # Draw per-document topic distribution
        theta = nprand.dirichlet(alpha)
        for i in range(DOC_LENGTH):
            topic = lda.sample(theta)
            word = lda.sample(beta[topic,:])
            corpus[m,word] += 1
    return corpus

if __name__ == '__main__':
    main()