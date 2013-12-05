'''
Generate data from a known model.
Perform both E and M step to find alpha and eta that maximize the likelihood.
'''

import os
import sys
import time

import numpy as np
import numpy.random as nprand
import scipy.misc as spmisc

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import lda

VOCAB_SIZE = 25
NUM_TOPICS = 10
NUM_DOCS = 100
DOC_LENGTH = 100
NUM_ITER = 25
GIBBS_BURN = 5
GIBBS_LAG = 5

alpha = 0.001*np.ones(NUM_TOPICS)
eta = 0.5*np.ones(VOCAB_SIZE)

id = time.time()

def main():
    # Create an image with one row per iteration to visualize results
    topic_img = np.zeros((NUM_ITER + 2, NUM_TOPICS*VOCAB_SIZE))
    print 'Generating corpus'
    corpus, beta = generate_corpus()
    # Add real topics to the image
    add_to_img(topic_img, NUM_ITER+1, beta, beta)
    alpha_guess = 0.001
    eta_guess = 0.5
    print 'Initializing model'
    model = lda.LdaModel(corpus, NUM_TOPICS, alpha_guess, eta_guess, GIBBS_BURN, GIBBS_LAG)
    add_to_img(topic_img, 0, beta, model.beta())
    # Do E-M iterations
    last_lik = model.log_likelihood_wz()
    for i in range(NUM_ITER):
        print 'Iteration %d' % i
        model.e_step()
        lik = model.log_likelihood_wz()
        print ' E-step ML: %f (%f)' % (lik, lik - last_lik)
        last_lik = lik
        model._m_alpha()
        lik = model.log_likelihood_wz()
        print ' M-step(alpha) ML: %f (%f)' % (lik, lik - last_lik)
        last_lik = lik
        model._m_eta()
        add_to_img(topic_img, i+1, beta, model.beta())
        lik = model.log_likelihood_wz()
        print ' M-step(eta) ML: %f (%f)' % (lik, lik - last_lik)
        last_lik = lik
    save_topics(topic_img)

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
    return (corpus, beta)

def add_to_img(img, i, real_beta, beta):
    '''Add model topics to an image.  Try to align with the real topics.'''
    ordered = np.zeros(beta.shape)
    beta = beta.copy()
    for j in range(beta.shape[0]):
        match = np.argmin(np.absolute(beta - real_beta[j]).sum(1))
        ordered[j,:] = beta[match,:]
        np.delete(beta, match, 0)
    img[i,:] = np.reshape(ordered, beta.shape[0] * beta.shape[1])
        

def save_topics(topic_img):
    try:
        os.stat('output')
    except OSError:
        os.mkdir('output')
    try:
        os.stat('output/demoldaem')
    except OSError:
        os.mkdir('output/demoldaem')
    filename = 'output/demoldaem/%s.png' % (id)
    topic_img = np.kron(topic_img, np.ones((5,5)))
    spmisc.imsave(filename, topic_img)

if __name__ == '__main__':
    main()