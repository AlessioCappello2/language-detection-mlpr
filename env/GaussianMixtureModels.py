import numpy
import scipy as sci
from GaussianModels import loglikelihood
from utils import *

def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    for g in range(len(gmm)):
        S[g, :] = loglikelihood(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])

    log_marginal_densities = vrow(sci.special.logsumexp(S, axis=0))
    return log_marginal_densities, S

def LBGalgorithm(X, components, alpha, psi, variant):
    mean = vcol(numpy.mean(X))
    gmm = [(1.0, mean, numpy.dot((X-mean), (X-mean).T) / X.shape[1])]
    while len(gmm) <= components:
        gmm = EMalgorithm(X, gmm, psi, variant)
        if len(gmm) == components:
            break
        gotgmm = []
        for i in range(len(gmm)):
            U, s, _ = numpy.linalg.svd(gmm[i][2])
            d = U[:, 0:1] * (s[0]**0.5) * alpha
            gotgmm.append((gmm[i][0]/2, gmm[i][1]+d, gmm[i][2]))
            gotgmm.append((gmm[i][0]/2, gmm[i][1]-d, gmm[i][2]))
        gmm = gotgmm
    return gmm

def EMalgorithm(X, gmm, psi, variant="Default", limit=10**(-6)):
    ll1 = None
    ll2 = None
    while ll1 is None or ll2-ll1 > limit:
        ll1 = ll2
        log_marginal_densities, S = logpdf_GMM(X, gmm)
        ll2 = numpy.sum(log_marginal_densities) / X.shape[1]
        # E step

        posterior = numpy.exp(S - vrow(log_marginal_densities))
        # M step
        gotgmm=[]
        commonCovMatrix = numpy.zeros((X.shape[0],X.shape[0]))
        for g in range(len(gmm)):
            gamma = posterior[g, :]
            Zg = numpy.sum(gamma)
            Fg = numpy.sum(vrow(gamma)*X, axis=1)
            Sg = numpy.dot(X, (vrow(gamma)*X).T)
            w = (Zg / posterior.sum())
            mu = (vcol(Fg / Zg))
            cov = ((Sg / Zg) - numpy.dot(mu, mu.T))
            if variant == 'Tied':
                commonCovMatrix += Zg*cov
                gotgmm.append((w, mu))
                continue
            if variant == 'Diagonal':
                cov *= numpy.eye(cov.shape[0])
            U, s, _ = numpy.linalg.svd(cov)
            s[s<psi] = psi
            cov = numpy.dot(U, vcol(s) * U.T)
            gotgmm.append((w, mu, cov))

        if variant == 'Tied':
            commonCovMatrix /= X.shape[1]
            U, s, _ = numpy.linalg.svd(commonCovMatrix)
            s[s<psi]= psi
            commonCovMatrix = numpy.dot(U, vcol(s)*U.T)
            gotgmmt = []
            for g in range(len(gotgmm)):
                (w, mu) = gotgmm[g]
                gotgmmt.append((w, mu, commonCovMatrix))
            gotgmm = gotgmmt
        gmm = gotgmm

    return gmm

def GMMclassify(DTR, LTR, DTE, LTE, prior, parameters, scores=True):
    variant = parameters[0]
    components = int(parameters[1])
    psi = float(parameters[2])
    alpha = float(parameters[3])
    
    D1 = DTR[:, LTR==1] 
    D0 = DTR[:, LTR==0]

    gmm1 = LBGalgorithm(D1, components, alpha, psi, variant)
    gmm0 = LBGalgorithm(D0, components, alpha, psi, variant)

    log_marginal_densities1, S1 = logpdf_GMM(DTE, gmm1)
    log_marginal_densities0, S0 = logpdf_GMM(DTE, gmm0)
    if scores:
        return numpy.array(log_marginal_densities1 - log_marginal_densities0).flatten()

    lmd = numpy.vstack((log_marginal_densities0, log_marginal_densities1))
    predictedLabels = numpy.argmax(lmd, axis=0)
    correctPredictions = numpy.array(predictedLabels == LTE).sum()
    accuracy = correctPredictions/LTE.size*100
    errorRate = 100.0 - accuracy
    return errorRate