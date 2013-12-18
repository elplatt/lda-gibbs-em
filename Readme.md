## Latent Dirichelet Allocation with Gibbs sampling

An LDA with Gibbs sampling implementation based on
"Parameter estimation for text analysis" by Gregor Heinrich.

Copyright 2013 MIT Center for Civic Media 
Distributed under BSD 3-Clause License, see LICENSE for info

## Dependencies

This library requires the following packages:
* numpy
* scipy

## How to use

For a detailed example see demos/demoldaem.py

First create a numpy array representing a term-document matrix with documents on dimension 0:

    corpus = numpy.array([[1,2,3], [4,5,6], [8,9,0], [9,8,7]])
    
Then create an LDA model, specifying the number of topics.  You can also specify the initial alpha (per-document Dirichlet parameter), eta (per-topic Dirichlet paramter).

    num_topics = 5
    alpha = 0.1
    eta = 0.1
    model = lda.LdaModel(corpus, num_topics, alpha, eta)
    
Then run a numer of EM iterations.

    model.em_iterate(25)

You can the query, for example, the expected per-topic word distributions (beta).

    beta = model.beta()
    print "Word distribution for topic 0: %s" % beta[0]
