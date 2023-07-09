import numpy
from utils import *

def primalObjective(w, DTR, LTR, C, fp):
    normTerm = 0.5*(numpy.linalg.norm(w)**2)
    z = numpy.zeros(vrow(LTR).shape[1])
    hingeLoss = numpy.maximum(z, 1-LTR*numpy.dot(w.T, DTR))
    primarLoss = normTerm + C*(hingeLoss.sum())
    dualLoss = -fp
    dualityGap = primarLoss - dualLoss
    return primarLoss, dualLoss, dualityGap


def LD_objectiveDualForm(alpha, H):
    return (0.5*numpy.dot(numpy.dot(alpha.transpose(), H), alpha) - numpy.dot(alpha.transpose(), vcol(numpy.ones(H.shape[1]))),
            numpy.dot(H, alpha) - numpy.ones(H.shape[1]))


def RBF(D1, D2, gamma):
    return numpy.exp(-gamma*(numpy.linalg.norm(D1.T[:, None] - D2.T, axis=-1))**2)


def SupportVectorMachineLinear(DTR, LTR, DTE, LTE, prior, parameters, scoresFlag=False):
    variant = parameters[0]
    K = float(parameters[1])
    C = float(parameters[2])
    DTR_cap = numpy.vstack([DTR, numpy.full(DTR.shape[1], K)])
    DTE_cap = numpy.vstack([DTE, numpy.full(DTE.shape[1], K)])
    ZTR = 2*LTR-1

    G = numpy.dot(DTR_cap.T, DTR_cap)
    H = numpy.dot(vcol(ZTR), vrow(ZTR)) * G

    constraints = [(0, C) for _ in range(DTR_cap.shape[1])]

    x, fp, d = sci.optimize.fmin_l_bfgs_b(LD_objectiveDualForm, numpy.zeros(DTR_cap.shape[1]), args=(H,), bounds=constraints, factr=1.0)

    w = numpy.sum(vrow(x*ZTR)*DTR_cap, axis=1)
    scores = numpy.dot(w.transpose(), DTE_cap)
    if scoresFlag:
        return scores

    predicted_labels = numpy.where(scores > 0, 1, 0)
    correctPrediction = numpy.array(predicted_labels == LTE).sum()
    errorRate = 100 - correctPrediction/LTE.size*100

    primar_loss, dual_loss, duality_gap = primalObjective(w, DTR_cap, ZTR, C, fp)
    print("K=%d, C=%f, Primal loss=%e, Dual loss=%e, Duality gap=%e, Error rate=%.1f %%" % (
        1, C, primar_loss, dual_loss, duality_gap, errorRate))


def SupportVectorMachineKernelPoly(DTR, LTR, DTE, LTE, prior, parameters, scoresFlag=False):
    variant = parameters[0]
    K = float(parameters[1])
    C = float(parameters[2])
    c = float(parameters[3])
    deg = int(parameters[4])
    ZTR = 2*LTR-1
    eps = K**2
    H = numpy.dot(vcol(ZTR), vrow(ZTR))*(((numpy.dot(DTR.T, DTR)+c)**deg)+eps)
    constraints = [(0, C) for _ in range(DTR.shape[1])]

    x, fp, d = sci.optimize.fmin_l_bfgs_b(LD_objectiveDualForm, numpy.zeros(DTR.shape[1]), args=(H,), bounds=constraints, factr=1.0)
    scores = numpy.sum(numpy.dot(vrow(x*ZTR), ((numpy.dot(DTR.T, DTE)+c)**deg+eps)), axis=0)
    if scoresFlag:
        return scores

    predicted_labels = numpy.where(scores > 0, 1, 0)

    # error rate
    correctPrediction = numpy.array(predicted_labels == LTE).sum()
    errorRate = 100 - correctPrediction/LTE.size*100
    print("Poly (d=%d, c=%d) | K = %.1f | C = %.1f | Dual loss = %e | Error rate = %.1f" % (
        d, c, K, C, -fp, errorRate))


def SupportVectorMachineKernelRBF(DTR, LTR, DTE, LTE, prior, parameters, scoresFlag=False):
    variant = parameters[0]
    K = float(parameters[1])
    C = float(parameters[2])
    gamma = float(parameters[3])
    ZTR = 2*LTR-1
    eps = K**2
    H = numpy.dot(vcol(ZTR), vrow(ZTR))*(RBF(DTR, DTR, gamma)+eps)
    constraints = [(0, C) for _ in range(DTR.shape[1])]

    x, fp, d = sci.optimize.fmin_l_bfgs_b(LD_objectiveDualForm, numpy.zeros(DTR.shape[1]), args=(H,), bounds=constraints, factr=1.0)
    scores = numpy.sum(numpy.dot(vrow(x*ZTR), (RBF(DTR, DTE, gamma)+eps)), axis=0)
    if scoresFlag:
        return scores

    predicted_labels = numpy.where(scores > 0, 1, 0)

    # error rate
    correctPrediction = numpy.array(predicted_labels == LTE).sum()
    errorRate = 100 - correctPrediction/LTE.size*100
    print("RBF (gamma=%.1f) | K = %.1f | C = %.1f | Dual loss = %e | Error rate = %.1f" % (
        gamma, K, C, -fp, errorRate))