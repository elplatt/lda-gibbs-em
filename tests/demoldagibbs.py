'''
Generate data from a known model and test LdaModel gibbs sampling on it.
The topic count and alpha are treated as known and no M-step is performed.
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

# Settings
TOPIC_ROWS = 5
TOPIC_COLS = 5
num_topics = TOPIC_ROWS + TOPIC_COLS
vocab_size = TOPIC_ROWS * TOPIC_COLS
alpha = np.ones(num_topics) / float(num_topics)
NUM_DOCS = 100
DOC_LENGTH = 100
ITERATIONS = 20
# Use timestamp to identify results
id = time.time()

def main():
    print 'Generating corpus'
    corpus = generate_corpus()
    print 'Initializing Model'
    model = lda.LdaModel(corpus, num_topics, alpha, 0, 0)
    print 'Sampling'
    for i in range(ITERATIONS):
        model._gibbs_sample(model.stats)
        print 'Iteration %d complete' % (i+1)
        save_beta(model.beta(), i)

def generate_corpus():
    corpus = np.zeros((NUM_DOCS, vocab_size), dtype='int64')
    beta = generate_beta()
    for m in range(NUM_DOCS):
        # Get topic distribution for current document
        theta = nprand.dirichlet(alpha)
        for i in range(DOC_LENGTH):
            # Sample topic
            zi = lda.sample(theta)
            w = lda.sample(beta[zi,:])
            corpus[m,w] += 1
    return corpus
    
def generate_beta():
    # Use two orthogonal sets of topics
    result = np.zeros((num_topics, vocab_size))
    # Create topics as rows/cols then reshape into array
    for k in range(TOPIC_ROWS):
        dist = np.zeros((TOPIC_ROWS, TOPIC_COLS))
        dist[k,:] = 1.0 / TOPIC_COLS
        result[k,:] = dist.reshape(vocab_size)
    for k in range(TOPIC_COLS):
        dist = np.zeros((TOPIC_ROWS, TOPIC_COLS))
        dist[:,k] = 1.0 / TOPIC_ROWS
        result[k + TOPIC_ROWS,:] = dist.reshape(vocab_size)
    return result

def save_beta(beta, i):
    try:
        os.stat('output')
    except OSError:
        os.mkdir('output')
    try:
        os.stat('output/demoldagibbs')
    except OSError:
        os.mkdir('output/demoldagibbs')
    try:
        os.stat('output/demoldagibbs/%s' % id)
    except OSError:
        os.mkdir('output/demoldagibbs/%s' % id)
    for k in range(beta.shape[0]):
        topic = beta[k,:].reshape((TOPIC_ROWS, TOPIC_COLS))
        filename = 'output/demoldagibbs/%s/iter%d-topic%d.png' % (id, i, k)
        img = np.kron(topic, np.ones((10,10)))
        spmisc.imsave(filename, img)

if __name__ == '__main__':
    main()