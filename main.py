from utils import *
import numpy
from DimensionalityReduction import *
from ModelValidation import *

def main():
    dataset_train, labels_train = load("./LanguageDetection/Train.txt")
    # dataset_test, labels_test = load("./LanguageDetection/Test.txt")
    k = 5
    workingPoint = (0.5, 1, 1)
    K = 0.0
    C = 100
    c = 1.0
    d = 2
    gamma = 0.01

    # pay attention to MVG_log and MVG
    classifiers = [(MVG_log, "Log-Multivariate Gaussian Classifier"), (MVG, "Naive Bayes Gaussian"),
                   (MVG, "Tied Covariance Gaussian"), (logisticRegression, "Logistic Regression"),
                   (logisticRegression, "Weighted Logistic Regression"), (logisticRegression, "Quadratic Logistic Regression"),
                   (logisticRegression, "Weighted Quadratic Logistic Regression"),
                   (SupportVectorMachineLinear, "Support Vector Machine - Linear"),
                   # (SupportVectorMachineKernelPoly, "Support Vector Machine - Kernel Poly"),
                   (SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")]

    parameters = [("Default"), ("Naive"), ("Tied"), ("Default", 0.001), ("Weighted", 0.001), ("Quadratic", 0.001),
                  ("Weighted quadratic", 0.001), ("SVML", K, C), ("SVMRBF", K, C, gamma)]
    '''("SVMP", K, C, c, d),'''

    classifiersLR = [(logisticRegression, "Logistic Regression"), (logisticRegression, "Weighted Logistic Regression"),
                     (logisticRegression, "Quadratic Logistic Regression"),
                     (logisticRegression, "Weighted Quadratic Logistic Regression")]

    parametersLR = [("Default", 0.001), ("Weighted", 0.001), ("Quadratic", 0.001), ("Weighted quadratic", 0.001)]
    #classifiersLR = [(logisticRegression, "Logistic Regression")]
    #parametersLR = [("Default", 0.001)]

    classifiersSVM = [(SupportVectorMachineLinear, "Support Vector Machine - Linear"),
                      (SupportVectorMachineKernelPoly, "Support Vector Machine - Kernel Poly"),
                      (SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")]

    parametersSVM = [("SVML", K, C), ("SVMP", K, C, c, d), ("SVMRBF", K, C, gamma)]

    classifiersGMM = [(GMMclassify, "GMM Full (1)"), (GMMclassify, "GMM Full (2)"), (GMMclassify, "GMM Full (4)"),
                      (GMMclassify, "GMM Full (8)"), (GMMclassify, "GMM Full (16)"), (GMMclassify, "GMM Full (32)"),
                      (GMMclassify, "GMM Diagonal (1)"), (GMMclassify, "GMM Diagonal (2)"), (GMMclassify, "GMM Diagonal (4)"),
                      (GMMclassify, "GMM Diagonal (8)"), (GMMclassify, "GMM Diagonal (16)"), (GMMclassify, "GMM Diagonal (32)"),
                      (GMMclassify, "GMM Tied (1)"), (GMMclassify, "GMM Tied (2)"), (GMMclassify, "GMM Tied (4)"),
                      (GMMclassify, "GMM Tied (8)"), (GMMclassify, "GMM Tied (16)"), (GMMclassify, "GMM Tied (32)")]

    parametersGMM = [("Default", 1, 0.01, 0.1), ("Default", 2, 0.01, 0.1), ("Default", 4, 0.01, 0.1),
                     ("Default", 8, 0.01, 0.1), ("Default", 16, 0.01, 0.1), ("Default", 32, 0.01, 0.1),
                     ("Diagonal", 1, 0.01, 0.1), ("Diagonal", 2, 0.01, 0.1), ("Diagonal", 4, 0.01, 0.1),
                     ("Diagonal", 8, 0.01, 0.1), ("Diagonal", 16, 0.01, 0.1), ("Diagonal", 32, 0.01, 0.1),
                     ("Tied", 1, 0.01, 0.1), ("Tied", 2, 0.01, 0.1), ("Tied", 4, 0.01, 0.1),
                     ("Tied", 8, 0.01, 0.1), ("Tied", 16, 0.01, 0.1), ("Tied", 32, 0.01, 0.1)]

    classifiersMVG = [(MVG_log, "Multivariate Gaussian Classifier"), (MVG_log, "Naive Bayes Gaussian"),
                      (MVG_log, "Tied Covariance Gaussian")]
    parametersMVG = [("Default"), ("Naive"), ("Tied")]

    # show_histo(dataset_train, labels_train)
    # show_histo(zNormalization(dataset_train), labels_train)

    heatmap(dataset_train[:, labels_train == 0], "Blues")
    heatmap(dataset_train[:, labels_train == 1], "Reds")
    heatmap(dataset_train, "Greys")
    minDCF = numpy.zeros(0)

    print(":------ MVG classifiers - NO PCA - 5 fold (0.5, 1, 1) ------:")
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, classifiersMVG, parametersMVG))
    print(":------ MVG classifiers -  PCA 6 - 5 fold (0.5, 1, 1) ------:")
    dataset = PCA(dataset_train, labels_train, 6, True)
    minDCF = numpy.append(minDCF, kfold(dataset, labels_train, k, workingPoint, classifiersMVG, parametersMVG))
    print(":------ MVG classifiers -  PCA 5 - 5 fold (0.5, 1, 1) ------:")
    dataset = PCA(dataset_train, labels_train, 5, True)
    minDCF = numpy.append(minDCF, kfold(dataset, labels_train, k, workingPoint, classifiersMVG, parametersMVG))
    print(":------ MVG classifiers -  PCA 4 - 5 fold (0.5, 1, 1) ------:")
    dataset = PCA(dataset_train, labels_train, 4, True)
    minDCF = numpy.append(minDCF, kfold(dataset, labels_train, k, workingPoint, classifiersMVG, parametersMVG))

    workingPoint = (0.1, 1, 1)
    print(":------ MVG classifiers - NO PCA - 5 fold (0.1, 1, 1) ------:")
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, classifiersMVG, parametersMVG))
    print(":------ MVG classifiers -  PCA 6 - 5 fold (0.1, 1, 1) ------:")
    dataset = PCA(dataset_train, labels_train, 6, True)
    minDCF = numpy.append(minDCF, kfold(dataset, labels_train, k, workingPoint, classifiersMVG, parametersMVG))
    print(":------ MVG classifiers -  PCA 5 - 5 fold (0.1, 1, 1) ------:")
    dataset = PCA(dataset_train, labels_train, 5, True)
    minDCF = numpy.append(minDCF, kfold(dataset, labels_train, k, workingPoint, classifiersMVG, parametersMVG))
    print(":------ MVG classifiers -  PCA 4 - 5 fold (0.5, 1, 1) ------:")
    dataset = PCA(dataset_train, labels_train, 4, True)
    minDCF = numpy.append(minDCF, kfold(dataset, labels_train, k, workingPoint, classifiersMVG, parametersMVG))

    print("eeeeeeeeeeee")
    print(minDCF)
    print(":----------------------- Average minDCF --------------------:")
    print("NO PCA - MVG: %f" % ((minDCF[0]+minDCF[12])/2))
    print("PCA 6  - MVG: %f" % ((minDCF[3]+minDCF[15])/2))
    print("PCA 5  - MVG: %f" % ((minDCF[6]+minDCF[18])/2))
    print("PCA 4  - MVG: %f" % ((minDCF[9]+minDCF[21])/2))
    print("NO PCA - Naive: %f" % ((minDCF[1]+minDCF[13])/2))
    print("PCA 6  - Naive: %f" % ((minDCF[4]+minDCF[16])/2))
    print("PCA 5  - Naive: %f" % ((minDCF[7]+minDCF[19])/2))
    print("PCA 4  - Naive: %f" % ((minDCF[10]+minDCF[22])/2))
    print("NO PCA - Tied: %f" % ((minDCF[2]+minDCF[14])/2))
    print("PCA 6  - Tied: %f" % ((minDCF[5]+minDCF[17])/2))
    print("PCA 5  - Tied: %f" % ((minDCF[8]+minDCF[20])/2))
    print("PCA 4  - Tied: %f" % ((minDCF[11]+minDCF[23])/2))
    # LDA(dataset_train, labels_train)


    # PCA(zNormalization(dataset_train), labels_train)
    # PCA(centerDataset(dataset_train), labels_train)
    # print("---------ORIGINAL DATA-------------\n")
    # kfold(dataset_train, labels_train, k, workingPoint, classifiers, parameters)
    # print("---------CENTERED DATA-------------\n")
    # kfold(centerDataset(dataset_train), labels_train, k, workingPoint, classifiers, parameters)
    # print("-------ZNORMALIZED DATA------------\n")
    # kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiers, parameters)
    # print("--------NORMALIZED DATA------------\n")
    # kfold(normalization(dataset_train), labels_train, k, workingPoint, classifiers, parameters)

    # kfold(dataset_train, labels_train, k, workingPoint, classifiersGMM, parametersGMM)

    # kfoldPlotMinDCFlambda(dataset_train, labels_train, k, workingPoint, classifiersLR, parametersLR)
    # kfoldPlotMinDCFC(dataset_train, labels_train, k, workingPoint, classifiersSVM, parametersSVM)
    # kfoldBayesErrorPlot(dataset_train, labels_train, k, workingPoint, logisticRegression, ("Quadratic", 0.001))
    # kfoldBayesErrorPlot(dataset_train, labels_train, k, workingPoint, SupportVectorMachineKernelRBF, ("SVMRBF", 1.0, 1.0, 10.0))

if __name__ == '__main__':
    main()
