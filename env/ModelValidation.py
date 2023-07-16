import numpy
import matplotlib.pyplot as plt
from DimensionalityReduction import PCA
from GaussianModels import *
from LogisticRegressionModels import *
from SupportVectorMachines import *
from GaussianMixtureModels import *
from BayesDecisions import *

def kfold(dataset, labels, k, workingPoint, classifiers, parameters, toCalibrate=False, plot=False, priorCalib=-1):
    K = k
    N = int(dataset.shape[1] / float(K))
    piT = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    array_mindcf = numpy.zeros(0)
    fusion_scores = []

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
            if parameters[j][0] == 'Weighted' or parameters[j][0] == 'Weighted quadratic': ## Prior weighted logistic regression
                #scoresfold.append(c(DTR, LTR, DTE, LTE, parameters[j][2], parameters[j], True, toCalibrate=toCalibrate))
                scoresfold.append(scoreCalibration(c(DTR, LTR, DTR, LTR, piT, parameters[j], True), LTR, c(DTR, LTR, DTE, LTE, piT, parameters[j], True, toCalibrate=toCalibrate), LTE, priorCalib if priorCalib else piT))
            else:
                scoresfold.append(c(DTR, LTR, DTE, LTE, priorCalib if priorCalib else piT, parameters[j], True, toCalibrate=toCalibrate))
            labelsfold.append(LTE)

        gotscores = numpy.hstack(scoresfold)
        gotlabels = numpy.hstack(labelsfold)

        cm = optimal_bayes_decisions(gotscores, gotlabels, workingPoint)
        DCFu = compute_bayes_risk(cm, workingPoint)
        actualDCF = DCFu/compute_dummy_bayes(workingPoint)
        minDCF = compute_minDCF(gotscores, gotlabels, workingPoint, plot)
        array_mindcf = numpy.append(array_mindcf, minDCF)
        # errorRate = nWrongPrediction / dataset.shape[1]
        # accuracy = 1 - errorRate
        # print(f"{cstring} results:\nAccuracy: {accuracy * 100}%\nError rate: {errorRate * 100}%\n")
        print(f"{cstring} results:\nActualDCF: {actualDCF}\nMinDCF: {minDCF}\n")
    '''if plot:
        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        plt.grid(True)
        plt.legend([classifiers[0][0], classifiers[1][0], classifiers[2][0]])
        plt.show()'''

    return array_mindcf


def kfoldBayesErrorPlot(dataset, labels, k, workingPoint, classifier, parameters, toCalibrate=False, color="r", plot=False):
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
        if parameters[0] == 'Weighted quadratic' and toCalibrate == True:
            scoresfold.append(scoreCalibration(classifier(DTR, LTR, DTR, LTR, piT, parameters, True), LTR, classifier(DTR, LTR, DTE, LTE, piT, parameters, True, toCalibrate=toCalibrate), LTE, 0.5))
        else:
            scoresfold.append(classifier(DTR, LTR, DTE, LTE, piT, parameters, True, toCalibrate=toCalibrate))
        labelsfold.append(LTE)

    gotscores = numpy.hstack(scoresfold)
    gotlabels = numpy.hstack(labelsfold)

    for i in range(pi_sign.size):
        cm = optimal_bayes_decisions(gotscores, gotlabels, (pi_sign[i], Cfn, Cfp))
        plot_dcf.append(compute_bayes_risk(cm, (pi_sign[i], Cfn, Cfp))/compute_dummy_bayes((pi_sign[i], 1, 1)))
        plot_mindcf.append(compute_minDCF(gotscores, gotlabels, (pi_sign[i], Cfn, Cfp)))

    plot_dcf = numpy.array(plot_dcf)
    plot_mindcf = numpy.array(plot_mindcf)

    if plot:
        return plot_dcf, plot_mindcf

    plt.plot(effPriorLogOdds, plot_dcf, linestyle="solid", color=color) #, label="DCF", color ="r")
    plt.plot(effPriorLogOdds, plot_mindcf, linestyle="dashed", color=color) #, label="minDCF", color ="b")
    # plt.ylim([0, max(plot_dcf)])
    plt.xlim([-3, 3])
    # plt.xlabel("prior log-odds")
    # plt.ylabel("minDCF")
    # plt.legend(["DCF", "minDCF"])
    # plt.show()


# Called ONLY with Logistic Regression in classifiers
def kfoldPlotMinDCFlambda(dataset, labels, k, workingPoints, classifiers, parameters):
    K = k
    N = int(dataset.shape[1] / float(K))
    '''piT1 = workingPoints[0][0]
    Cfn1 = workingPoints[0][1]
    Cfp1 = workingPoints[0][2]'''
    lambda_r = numpy.logspace(-5, 3, num=9)
    colors = ("r", "b", "g", "m")
    plot_mindcf = []

    for w in range(len(workingPoints)):
        piT = workingPoints[w][0]
        Cfn = workingPoints[w][1]
        Cfp = workingPoints[w][2]
        for j, (c, cstring) in enumerate(classifiers):
            color = colors[j]
            nWrongPrediction = 0
            # scoresfold = []
            # labelsfold = []

            splits = split_db_k(dataset.shape[1], K, seed=0)
            zeroTon = numpy.arange(0, dataset.shape[1])
            # plot_mindcf = []

            for l in range(lambda_r.size):
                # par = (parameters[j][0], l)
                scoresfold = []
                labelsfold = []
                par = (parameters[j][0], lambda_r[l])
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

                cm = optimal_bayes_decisions(gotscores, gotlabels, (piT, Cfp, Cfn))
                plot_mindcf.append(compute_minDCF(gotscores, gotlabels, (piT, Cfp, Cfn)))

    # print(plot_mindcf)
    plot_mindcf = [(plot_mindcf[i]+plot_mindcf[lambda_r.size+i])/2 for i in range (lambda_r.size)]
    # print(plot_mindcf)
    # print(lambda_r)
    # print(plot_mindcf)
    plt.plot(lambda_r, plot_mindcf, label="λ")#, color = color)
    '''plt.xlabel("λ")
    plt.ylabel("minCprim")
    plt.xscale('log')
    plt.ylim([min(plot_mindcf), max(plot_mindcf)])
    plt.xlim([10**-5, 10**3])
    # plt.legend(["Default LR", "Weighted LR", "Quadratic LR", "Weighted Quad. LR"])
    plt.legend(["Log-Reg", "Log-Reg (z-norm)"])
    plt.show()'''


# Called ONLY with SVM in classifiers
def kfoldPlotMinDCFC(dataset, labels, k, workingPoints, classifiers, parameters):
    K = k
    N = int(dataset.shape[1] / float(K))
    #piT = workingPoint[0]
    #Cfn = workingPoint[1]
    #Cfp = workingPoint[2]
    C = numpy.logspace(-5, 2, num=8)
    colors = ("r", "b", "g", "m")
    plot_mindcf = []

    #for w in range(len(workingPoints)):
    #    piT = workingPoints[w][0]
    #    Cfn = workingPoints[w][1]
    #    Cfp = workingPoints[w][2]
    for j, (c, cstring) in enumerate(classifiers):
        color = colors[j]
        nWrongPrediction = 0
        #scoresfold = []
        #labelsfold = []

        splits = split_db_k(dataset.shape[1], K, seed=0)
        zeroTon = numpy.arange(0, dataset.shape[1])
        #plot_mindcf = []

        for l in range(C.size):
            scoresfold = []
            labelsfold = []
            if parameters[j][0] == "SVML":
                par = (parameters[j][0], parameters[j][1], C[l])
            elif parameters[j][0] == "SVMP":
                par = (parameters[j][0], parameters[j][1], C[l], parameters[j][3], parameters[j][4])
            elif parameters[j][0] == "SVMRBF":
                par = (parameters[j][0], parameters[j][1], C[l], parameters[j][3])
            for i in range(K):
                idx = numpy.setdiff1d(zeroTon, splits[i])

                DTR = dataset[:, idx]
                LTR = labels[idx]
                DTE = dataset[:, splits[i]]
                LTE = labels[splits[i]]
            # nCorrectPrediction, nSamples = c(DTR, LTR, DTE, LTE, prior, True)
            # nWrongPrediction += nSamples - nCorrectPrediction
                # 1 -> piT
                scoresfold.append(c(DTR, LTR, DTE, LTE, 1, par, True))
                labelsfold.append(LTE)

            gotscores = numpy.hstack(scoresfold)
            gotlabels = numpy.hstack(labelsfold)

            #cm = optimal_bayes_decisions(gotscores, gotlabels, (piT, 1, 1))
            plot_mindcf.append(compute_minDCF(gotscores, gotlabels, (workingPoints[0][0], workingPoints[0][1], workingPoints[0][2])))
            plot_mindcf.append(compute_minDCF(gotscores, gotlabels, (workingPoints[1][0], workingPoints[1][1], workingPoints[1][2])))

    plot_mindcf = [(plot_mindcf[i]+plot_mindcf[i+1])/2 for i in range (0, 2*C.size, 2)]
    plt.plot(C, plot_mindcf, label="C", linestyle="dashed")#, color=color)

    '''plt.xscale('log')
    plt.ylim([min(plot_mindcf), max(plot_mindcf)])
    plt.xlim([10 ** -4, 10 ** -2])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend(["Linear SVM", "KernelPoly SVM (d=%d, c=%.1f)"%(parameters[1][4], parameters[1][3]), 
                "KernelRBF SVM (gamma=%.1f)"%(parameters[2][3])])
    plt.show()'''


def kfoldFusion(dataset, labels, k, workingPoint, classifiers, parameters, toCalibrate, pca=[], plot=False):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    pi_sign = 1/(1+numpy.exp(-effPriorLogOdds))
    K = k
    N = int(dataset.shape[1] / float(K))
    piT = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]
    array_mindcf = numpy.zeros(0)
    fusion_scores = []
    fusion_labels = []
    plot_mindcf = []
    plot_dcf = []

    splits = split_db_k(dataset.shape[1], K, seed=0)
    zeroTon = numpy.arange(0, dataset.shape[1])

    for i in range(K):
        scoresfold = []
        scoresev = []
        idx = numpy.setdiff1d(zeroTon, splits[i])

        #DTR = dataset[:, idx]
        #LTR = labels[idx]
        #DTE = dataset[:, splits[i]]
        #LTE = labels[splits[i]]
        for j, (c, cstring) in enumerate(classifiers):
            _dataset = dataset if pca[j] == -1 else PCA(dataset, labels, int(pca[j]))
            DTR = _dataset[:, idx]
            LTR = labels[idx]
            DTE = _dataset[:, splits[i]]
            LTE = labels[splits[i]]
            if parameters[j][0] == 'Weighted' or parameters[j][0] == 'Weighted quadratic': ## Prior weighted logistic regression
                scoresfold.append(c(DTR, LTR, DTR, LTR, parameters[j][2], parameters[j], True, toCalibrate=toCalibrate[j])) ## in caso togliere
                scoresev.append(c(DTR, LTR, DTE, LTE, parameters[j][2], parameters[j], True, toCalibrate=toCalibrate[j]))
            else:
                scoresfold.append(c(DTR, LTR, DTR, LTR, piT, parameters[j], True, toCalibrate=toCalibrate[j])) ## in caso togliere
                scoresev.append(c(DTR, LTR, DTE, LTE, piT, parameters[j], True, toCalibrate=toCalibrate[j]))

        gotscores = numpy.vstack(scoresfold)
        gotev = numpy.vstack(scoresev)

        fusion_labels.append(LTE)
        fusion_scores.append(modelFusion(gotscores, LTR, gotev, LTE, 0.5))

    fusion_labels = numpy.hstack(fusion_labels)
    fusion_scores = numpy.hstack(fusion_scores)

    cm = optimal_bayes_decisions(fusion_scores, fusion_labels, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF = DCFu / compute_dummy_bayes(workingPoint)
    minDCF = compute_minDCF(fusion_scores, fusion_labels, workingPoint)

    print(f"Fusion results:\nActualDCF: {actualDCF}\nMinDCF: {minDCF}\n")

    #print(fusion_labels.shape)
    #print(fusion_scores.shape)

    for i in range(pi_sign.size):
        cm = optimal_bayes_decisions(fusion_scores, fusion_labels, (pi_sign[i], Cfn, Cfp))
        plot_dcf.append(compute_bayes_risk(cm, (pi_sign[i], Cfn, Cfp))/compute_dummy_bayes((pi_sign[i], 1, 1)))
        plot_mindcf.append(compute_minDCF(fusion_scores, fusion_labels, (pi_sign[i], Cfn, Cfp)))

    plot_dcf = numpy.array(plot_dcf)
    plot_mindcf = numpy.array(plot_mindcf)

    if plot:
        return plot_dcf, plot_mindcf

    plt.plot(effPriorLogOdds, plot_dcf, linestyle="solid", color="y") #, label="DCF", color ="r")
    plt.plot(effPriorLogOdds, plot_mindcf, linestyle="dashed", color="y") #, label="minDCF", color ="b")
    plt.xlim([-3, 3])


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