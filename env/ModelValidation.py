import numpy
import matplotlib.pyplot as plt
from GaussianModels import *
from LogisticRegressionModels import *
from SupportVectorMachines import *
from BayesDecisions import *

def kfold(dataset, labels, k, workingPoint, classifiers, parameters):
    K = k
    N = int(dataset.shape[1] / float(K))
    piT = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]

    for j, (c, cstring) in enumerate(classifiers):
        nWrongPrediction = 0
        scoresfold = []
        labelsfold = []

        splits = split_db_k(dataset.shape[1], K, seed=0)
        zeroTon = numpy.arange(0, dataset.shape[1])

        for i in range(K):

            idx = numpy.setdiff1d(zeroTon, splits[i])

            DTR = dataset[:, idx]
            LTR = labels[idx]
            DTE = dataset[:, splits[i]]
            LTE = labels[splits[i]]
            # nCorrectPrediction, nSamples = c(DTR, LTR, DTE, LTE, prior, True)
            # nWrongPrediction += nSamples - nCorrectPrediction
            scoresfold.append(c(DTR, LTR, DTE, LTE, piT, parameters[j], True))
            labelsfold.append(LTE)

        gotscores = numpy.hstack(scoresfold)
        gotlabels = numpy.hstack(labelsfold)

        cm = optimal_bayes_decisions(gotscores, gotlabels, workingPoint)
        print(cm)
        DCFu = compute_bayes_risk(cm, workingPoint)
        actualDCF = DCFu/compute_dummy_bayes(workingPoint)
        minDCF = compute_minDCF(gotscores, gotlabels, workingPoint)

        # errorRate = nWrongPrediction / dataset.shape[1]
        # accuracy = 1 - errorRate
        # print(f"{cstring} results:\nAccuracy: {accuracy * 100}%\nError rate: {errorRate * 100}%\n")
        print(f"{cstring} results:\nActualDCF: {actualDCF}\nMinDCF: {minDCF}\n")

def kfoldBayesErrorPlot(dataset, labels, k, workingPoint, classifier, parameters):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    pi_sign = 1/(1+numpy.exp(-effPriorLogOdds))
    plot_dcf = []
    plot_mindcf = []
    K = k
    N = int(dataset.shape[1] / float(K))
    piT = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]

    scoresfold = []
    labelsfold = []

    splits = split_db_k(dataset.shape[1], K, seed=0)
    zeroTon = numpy.arange(0, dataset.shape[1])

    for i in range(K):
        idx = numpy.setdiff1d(zeroTon, splits[i])

        DTR = dataset[:, idx]
        LTR = labels[idx]
        DTE = dataset[:, splits[i]]
        LTE = labels[splits[i]]
        # nCorrectPrediction, nSamples = c(DTR, LTR, DTE, LTE, prior, True)
        # nWrongPrediction += nSamples - nCorrectPrediction
        scoresfold.append(classifier(DTR, LTR, DTE, LTE, piT, parameters, True))
        labelsfold.append(LTE)

    gotscores = numpy.hstack(scoresfold)
    gotlabels = numpy.hstack(labelsfold)

    for i in range(pi_sign.size):
        cm = optimal_bayes_decisions(gotscores, gotlabels, (pi_sign[i], Cfn, Cfp))
        plot_dcf.append(compute_bayes_risk(cm, (pi_sign[i], Cfn, Cfp))/compute_dummy_bayes((pi_sign[i], 1, 1)))
        plot_mindcf.append(compute_minDCF(gotscores, gotlabels, (pi_sign[i], Cfn, Cfp)))

    plot_dcf = numpy.array(plot_dcf)
    plot_mindcf = numpy.array(plot_mindcf)

    plt.plot(effPriorLogOdds, plot_dcf, label="DCF", color ="r")
    plt.plot(effPriorLogOdds, plot_mindcf, label="minDCF", color ="b")
    plt.ylim([0, max(plot_dcf)])
    plt.xlim([-3, 3])
    plt.xlabel("prior log-odds")
    plt.ylabel("minDCF")
    plt.legend(["DCF", "minDCF"])
    plt.show()


# Called ONLY with Logistic Regression in classifiers
def kfoldPlotMinDCFlambda(dataset, labels, k, workingPoint, classifiers, parameters):
    K = k
    N = int(dataset.shape[1] / float(K))
    piT = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    lambda_r = numpy.linspace(10**-5, 10**3, 21)
    colors = ("r", "b", "g", "m")

    for j, (c, cstring) in enumerate(classifiers):
        color = colors[j]
        nWrongPrediction = 0
        scoresfold = []
        labelsfold = []

        splits = split_db_k(dataset.shape[1], K, seed=0)
        zeroTon = numpy.arange(0, dataset.shape[1])
        plot_mindcf = []

        for l in range(lambda_r.size):
            par = (parameters[j][0], l)
            for i in range(K):
                idx = numpy.setdiff1d(zeroTon, splits[i])

                DTR = dataset[:, idx]
                LTR = labels[idx]
                DTE = dataset[:, splits[i]]
                LTE = labels[splits[i]]
                # nCorrectPrediction, nSamples = c(DTR, LTR, DTE, LTE, prior, True)
                # nWrongPrediction += nSamples - nCorrectPrediction
                scoresfold.append(c(DTR, LTR, DTE, LTE, piT, par, True))
                labelsfold.append(LTE)

            gotscores = numpy.hstack(scoresfold)
            gotlabels = numpy.hstack(labelsfold)

            cm = optimal_bayes_decisions(gotscores, gotlabels, (piT, 1, 1))
            plot_mindcf.append(compute_minDCF(gotscores, gotlabels, (piT, 1, 1)))

        plt.plot(lambda_r, plot_mindcf, label="λ", color = color)
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.xscale('log')
    plt.ylim([min(plot_mindcf), max(plot_mindcf)])
    plt.xlim([10**-5, 10**3])
    plt.legend(["Default LR", "Weighted LR", "Quadratic LR", "Weighted Quad. LR"])
    plt.show()


# Called ONLY with SVM in classifiers
def kfoldPlotMinDCFC(dataset, labels, k, workingPoint, classifiers, parameters):
    K = k
    N = int(dataset.shape[1] / float(K))
    piT = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    C = numpy.linspace(10**-4, 10**-2, 21)
    colors = ("r", "b", "g", "m")

    for j, (c, cstring) in enumerate(classifiers):
        color = colors[j]
        nWrongPrediction = 0
        scoresfold = []
        labelsfold = []

        splits = split_db_k(dataset.shape[1], K, seed=0)
        zeroTon = numpy.arange(0, dataset.shape[1])
        plot_mindcf = []

        for l in range(lambda_r.size):
            if parameters[j][0] == "SVML":
                par = (parameters[j][0], parameters[j][1], l)
            elif parameters[j][0] == "SVMP":
                par = (parameters[j][0], parameters[j][1], l, parameters[j][3], parameters[j][4])
            elif parameters[j][0] == "SVMRBF":
                par = (parameters[j][0], parameters[j][1], l, parameters[j][3])
            for i in range(K):
                idx = numpy.setdiff1d(zeroTon, splits[i])

                DTR = dataset[:, idx]
                LTR = labels[idx]
                DTE = dataset[:, splits[i]]
                LTE = labels[splits[i]]
                # nCorrectPrediction, nSamples = c(DTR, LTR, DTE, LTE, prior, True)
                # nWrongPrediction += nSamples - nCorrectPrediction
                scoresfold.append(c(DTR, LTR, DTE, LTE, piT, par, True))
                labelsfold.append(LTE)

            gotscores = numpy.hstack(scoresfold)
            gotlabels = numpy.hstack(labelsfold)

            cm = optimal_bayes_decisions(gotscores, gotlabels, (piT, 1, 1))
            plot_mindcf.append(compute_minDCF(gotscores, gotlabels, (piT, 1, 1)))

        plt.plot(lambda_r, plot_mindcf, label="C", color=color)

    plt.xscale('log')
    plt.ylim([min(plot_mindcf), max(plot_mindcf)])
    plt.xlim([10 ** -4, 10 ** -2])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend(["Linear SVM", "KernelPoly SVM (d=%d, c=%.1f)"%(parameters[1][4], parameters[1][3]), 
                "KernelRBF SVM (gamma=%.1f)"%(parameters[2][3])])
    plt.show()


def split_db_k(n, k, seed=0):
    nFold = int(n*1.0/k)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(n)
    res = []
    res.append(idx[0:nFold])
    for i in range(1, k-1):
        res.append(idx[i * nFold:(i + 1) * nFold])
    res.append(idx[(k - 1) * nFold:])

    return res