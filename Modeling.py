from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from hmmlearn import hmm
from numpy.random import dirichlet
import numpy as np

def kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    labels = labels.reshape((len(labels)))
    return labels

def GMMmodel(data, n_mix):
    gmm = GMM(n_components=n_mix)
    labels = gmm.fit_predict(data)
    labels = labels.reshape((len(labels)))
    return labels

def HMMmodel(data, n_emissions, n_states, ):
    transmat = dirichlet([3]*n_states, n_states)
    emitmat = dirichlet([3]*n_emissions, n_states)
    startprob = dirichlet([3]*n_states)

    h = hmm.MultinomialHMM(n_components=n_states)
    h.startprob=startprob
    h.transmat=transmat
    h.emissionprob_ = emitmat

    h.fit(data)
    return h
