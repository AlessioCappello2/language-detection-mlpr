import numpy
import scipy as sci

def features_expansion(D):
    expansion = []
    for i in range(D.shape[1]):
        vec = numpy.reshape(numpy.dot(vcol(D[:, i]), vcol(D[:, i]).T), (-1, 1), order='F')
        expansion.append(vec)
    return numpy.vstack((numpy.hstack(expansion), D))


def J(w, b, DTR, LTR, lambda_r, prior):
    z = (LTR*2) - 1
    norm_term = 0.5 * lambda_r * (numpy.linalg.norm(w) ** 2)
    if prior >= 0:
        c1 = ((prior) / (LTR[LTR == 1].shape[0])) * numpy.logaddexp(0, -z[z==1]*(numpy.dot(w.T, dtr[:, LTR == 1])+b)).sum()
        c0 = ((1-prior) / (LTR[LTR == 0].shape[0])) * numpy.logaddexp(0, -z[z==-1]*(numpy.dot(w.T, dtr[:, LTR == 0])+b)).sum()
        return norm_term + c1 + c0
    else:
        c = (LTR.shape[0] ** -1) * numpy.logaddexp(0, -z*numpy.dot(w.T, DTR[:, :])+b)
        return norm_term + c

def logreg_obj_wrap(dtr, ltr, lambda_r, prior=-1):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        return J(w, b, dtr, ltr, lambda_r, prior)
    return logreg_obj


def logisticRegression(DTR, LTR, DTE, LTE, prior, variant='Default', lambda_r=1, score=False):
    if variant == 'Default':
        x, fp, d = sci.optimize.fmin_l_bfgs_b(logreg_obj_wrap(DTR, LTR, lambda_r), numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    else: 
        x, fp, d = sci.optimize.fmin_l_bfgs_b(logreg_obj_wrap(DTR, LTR, lambda_r, prior))    
        
    scores = numpy.dot(x[0:-1], DTE)+x[-1]
    if score:
        return scores
    
    predicted_labels = numpy.where(scores > 0, 1, 0)
    return numpy.count_nonzero(predicted_labels - LTE == 0), LTE.size