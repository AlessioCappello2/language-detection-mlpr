import numpy

def kfold(dataset, labels, k, workingPoint):
    K = k
    N = int(dataset.shape[1] / float(K))
    piT = workingPoint[0]
    Cfn = workingPoint[1]
    Cfp = workingPoint[2]

    classifiers = [(MVG_log, "Multivariate Gaussian Classifier"), (MVG_NaiveBayes, "Naive Bayes"),
                   (MVG_TiedCovariance, "Tied Covariance"), (logisticRegression, "Logistic Regression"),
                   (smartLogisticRegression, "Smart Logistic Regression"), (quadraticLogisticRegression, "Quadratic Logistic Regression")]

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
            scoresfold.append(c(DTR, LTR, DTE, LTE, piT, True))
            labelsfold.append(LTE)

        gotscores = numpy.hstack(scoresfold)
        gotlabels = numpy.hstack(labelsfold)

        cm = optimal_bayes_decisions(gotscores, gotlabels, workingPoint)
        DCFu = compute_bayes_risk(cm, workingPoint)
        actualDCF = DCFu /compute_dummy_bayes(workingPoint)
        minDCF = compute_minDCF(gotscores, gotlabels, workingPoint)

        # errorRate = nWrongPrediction / dataset.shape[1]
        # accuracy = 1 - errorRate
        # print(f"{cstring} results:\nAccuracy: {accuracy * 100}%\nError rate: {errorRate * 100}%\n")
        print(f"{cstring} results:\nActualDCF: {actualDCF}\nMinDCF: {minDCF}\n")


def split_db_k(n, k, seed=0):
    nFold = int(n*1.0/k)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(n)
    res = []
    res.append(idx[0:nFold])
    for i in range(1, k- 1):
        res.append(idx[i * nFold:(i + 1) * nFold])
    res.append(idx[(k - 1) * nFold:])

    return res