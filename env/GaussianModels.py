import numpy
import scipy as sci
from utils import *

def compute_means_cov_matrixes(d, l):
    dataset_classes = []
    means = []
    cov_matrixes = []

    for i in range(2):
        dataset_classes.append(d[:, l == i])
        means.append(dataset_classes[i].mean(1))
        cov_matrixes.append(numpy.dot(dataset_classes[i]-vcol(means[i]), (dataset_classes[i]-vcol(means[i])).transpose())/dataset_classes[i].shape[1])

    return means, cov_matrixes


def logpdf_GAU_ND(x, mu, cov):
    nd = x.shape[0]
    inv_cov = numpy.linalg.inv(cov)
    det = numpy.linalg.slogdet(cov)[1]
    log_density = (-nd/2) * numpy.log(2 * numpy.pi)
    log_density += - 0.5 * det
    less_mean = x - vcol(mu)
    log_density += - 0.5 * (less_mean * numpy.dot(inv_cov, less_mean)).sum(0)
    return log_density


def loglikelihood(x_test, mu_class, cov_class):
    return logpdf_GAU_ND(x_test, mu_class, cov_class)


def MVG_log(DTR, LTR, DTE, LTE, prior, parameters, scores=False):
    variant = parameters
    class_means, class_cov_matrixes = compute_means_cov_matrixes(DTR, LTR)
    ll_classes = []
    if variant == 'Tied':
        common_covariance_matrix = numpy.zeros([DTR.shape[0], DTR.shape[0]])
        for i in range(2):
            common_covariance_matrix += class_cov_matrixes[i] * float(numpy.count_nonzero(LTR == i))
        common_covariance_matrix /= LTR.size
    
    for i in range(2):
        if variant == 'Default':
            ll_classes.append(loglikelihood(DTE, class_means[i], class_cov_matrixes[i]))
        elif variant == 'Naive':
            ll_classes.append(loglikelihood(DTE, class_means[i], class_cov_matrixes[i] * numpy.identity(DTR.shape[0])))
        elif variant == 'Tied':
            ll_classes.append(loglikelihood(DTE, class_means[i], common_covariance_matrix))

    log_class_conditional_densities = numpy.array(ll_classes)
    if scores:
        c1 = numpy.array(log_class_conditional_densities[1])
        c0 = numpy.array(log_class_conditional_densities[0])
        return c1 - c0
    
    # joint_densities = numpy.multiply(class_conditional_densities, vcol(numpy.array([prior, 1-prior])))  # fx,c = fx|c(xt | c) * P(C)
    # marginal_densities = vrow(joint_densities.sum(0))  # sum fx,c over c
    # post_conditional_prob = joint_densities / marginal_densities
    log_joint_densities = log_class_conditional_densities + numpy.log(vcol(numpy.array([prior, 1-prior])))
    log_marginal_densities = vrow(sci.special.logsumexp(log_joint_densities, axis=0))
    log_post_conditional_prob = log_joint_densities - log_marginal_densities
    post_conditional_prob = numpy.exp(log_post_conditional_prob)

    predicted_labels = numpy.argmax(post_conditional_prob, axis=0)
    nblog_error_rate = numpy.count_nonzero(predicted_labels-LTE)/predicted_labels.shape[0]
    return numpy.count_nonzero(predicted_labels-LTE == 0), LTE.size


def MVG(DTR, LTR, DTE, LTE, prior, parameters, scores=False):
    variant = parameters
    class_means, class_cov_matrixes = compute_means_cov_matrixes(DTR, LTR)
    ll_classes = []
    if variant == 'Tied':
        common_covariance_matrix = numpy.zeros([6, 6])
        for i in range(2):
            common_covariance_matrix += class_cov_matrixes[i] * float(numpy.count_nonzero(LTR == i))
        common_covariance_matrix /= LTR.size

    for i in range(2):
        if variant == 'Default':
            ll_classes.append(loglikelihood(DTE, class_means[i], class_cov_matrixes[i]))
        elif variant == 'Naive':
            ll_classes.append(loglikelihood(DTE, class_means[i], class_cov_matrixes[i] * numpy.identity(6)))
        elif variant == 'Tied':
            ll_classes.append(loglikelihood(DTE, class_means[i], common_covariance_matrix))

    class_conditional_densities = numpy.exp(numpy.array(ll_classes))  # fx|c(xt| c)
    if scores:
        c1 = numpy.array(class_conditional_densities[1])
        c0 = numpy.array(class_conditional_densities[0])
        return c1-c0
    
    joint_densities = numpy.multiply(class_conditional_densities, vcol(numpy.array([prior, 1-prior])))  # fx,c = fx|c(xt | c) * P(C)
    marginal_densities = vrow(joint_densities.sum(0))  # sum fx,c over c
    post_conditional_prob = joint_densities / marginal_densities

    predicted_labels = numpy.argmax(post_conditional_prob, axis=0)
    log_error_rate = numpy.count_nonzero(predicted_labels-lte)/predicted_labels.shape[0]
    # print(nblog_error_rate)
    return numpy.count_nonzero(predicted_labels-LTE == 0), LTE.size