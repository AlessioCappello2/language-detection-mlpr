from utils import *
import numpy
import matplotlib.pyplot as plt
import sys
from DimensionalityReduction import *
from ModelValidation import *

def main():
    # path = 'LanguageDetection/EVAL_GMM.txt'
    # path = 'LanguageDetection/LogisticRegressionFixedPrior2(100)(0.2).txt'
    # sys.stdout = open(path, 'w')

    dataset_train, labels_train = load("./LanguageDetection/Train.txt")
    dataset_test, labels_test = load("./LanguageDetection/Test.txt")
    k = 5
    workingPoint = (0.5, 1, 1)
    K = 1.0
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

    parametersLR = [("Default", 100), ("Weighted", 100, 0.2), ("Quadratic", 100), ("Weighted quadratic", 100, 0.2)]
    parametersLR0 = [("Default", 0), ("Weighted", 0), ("Quadratic", 0), ("Weighted quadratic", 0)]
    #classifiersLR = [(logisticRegression, "Logistic Regression")]
    #parametersLR = [("Default", 0.001)]

    classifiersSVM = [(SupportVectorMachineLinear, "Support Vector Machine - Linear"),
                      (SupportVectorMachineKernelPoly, "Support Vector Machine - Kernel Poly"),
                      (SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")]

    parametersSVM = [("SVML", K, C), ("SVMP", K, C, c, d), ("SVMRBF", K, C, gamma)]

    classifiersGMM = [(GMMclassify, "GMM Full T1 - Diagonal NT1)"), (GMMclassify, "GMM Full T1 - Diagonal NT2"), (GMMclassify, "GMM Full T1 - Diagonal NT4"),
                      (GMMclassify, "GMM Full T1 - Diagonal NT8)"), (GMMclassify, "GMM Full T1 - Diagonal NT16"), (GMMclassify, "GMM Full T1 - Diagonal NT32"),
                      (GMMclassify, "GMM Full T1 - Diagonal Tied NT1)"), (GMMclassify, "GMM Full T1 - Diagonal Tied NT2"), (GMMclassify, "GMM Full T1 - Diagonal Tied NT4"),
                      (GMMclassify, "GMM Full T1 - Diagonal Tied NT8)"), (GMMclassify, "GMM Full T1 - Diagonal Tied NT16"), (GMMclassify, "GMM Full T1 - Diagonal Tied NT32"),
                      (GMMclassify, "GMM Full T2 - Diagonal NT1)"), (GMMclassify, "GMM Full T2 - Diagonal NT2"), (GMMclassify, "GMM Full T2 - Diagonal NT4"),
                      (GMMclassify, "GMM Full T2 - Diagonal NT8)"), (GMMclassify, "GMM Full T2 - Diagonal NT16"), (GMMclassify, "GMM Full T2 - Diagonal NT32"),
                      (GMMclassify, "GMM Full T2 - Diagonal Tied NT1)"), (GMMclassify, "GMM Full T2 - Diagonal Tied NT2"), (GMMclassify, "GMM Full T2 - Diagonal Tied NT4"),
                      (GMMclassify, "GMM Full T2 - Diagonal Tied NT8)"), (GMMclassify, "GMM Full T2 - Diagonal Tied NT16"), (GMMclassify, "GMM Full T2 - Diagonal Tied NT32"),
                      (GMMclassify, "GMM Full T4 - Diagonal NT1)"), (GMMclassify, "GMM Full T4 - Diagonal NT2"), (GMMclassify, "GMM Full T4 - Diagonal NT4"),
                      (GMMclassify, "GMM Full T4 - Diagonal NT8)"), (GMMclassify, "GMM Full T4 - Diagonal NT16"), (GMMclassify, "GMM Full T4 - Diagonal NT32"),
                      (GMMclassify, "GMM Full T4 - Diagonal Tied NT1)"), (GMMclassify, "GMM Full T4 - Diagonal Tied NT2"), (GMMclassify, "GMM Full T4 - Diagonal Tied NT4"),
                      (GMMclassify, "GMM Full T4 - Diagonal Tied NT8)"), (GMMclassify, "GMM Full T4 - Diagonal Tied NT16"), (GMMclassify, "GMM Full T4 - Diagonal Tied NT32"),

                      (GMMclassify, "GMM Diagonal T1 - Full NT1"), (GMMclassify, "GMM Diagonal T1 - Full NT2"), (GMMclassify, "GMM Diagonal T1 - Full NT4"),
                      (GMMclassify, "GMM Diagonal T1 - Full NT8"), (GMMclassify, "GMM Diagonal T1 - Full NT16"), (GMMclassify, "GMM Diagonal T1 - Full NT32"),
                      (GMMclassify, "GMM Diagonal T1 - Diagonal Tied NT1)"), (GMMclassify, "GMM Diagonal T1 - Diagonal Tied NT2"), (GMMclassify, "GMM Diagonal T1 - Diagonal Tied NT4"),
                      (GMMclassify, "GMM Diagonal T1 - Diagonal Tied NT8)"), (GMMclassify, "GMM Diagonal T1 - Diagonal Tied NT16"), (GMMclassify, "GMM Diagonal T1 - Diagonal Tied NT32"),
                      (GMMclassify, "GMM Diagonal T2 - Full NT1)"), (GMMclassify, "GMM Diagonal T2 - Full NT2"), (GMMclassify, "GMM Diagonal T2 - Full NT4"),
                      (GMMclassify, "GMM Diagonal T2 - Full NT8)"), (GMMclassify, "GMM Diagonal T2 - Full NT16"), (GMMclassify, "GMM Diagonal T2 - Full NT32"),
                      (GMMclassify, "GMM Diagonal T2 - Diagonal Tied NT1)"), (GMMclassify, "GMM Diagonal T2 - Diagonal Tied NT2"), (GMMclassify, "GMM Diagonal T2 - Diagonal Tied NT4"),
                      (GMMclassify, "GMM Diagonal T2 - Diagonal Tied NT8)"), (GMMclassify, "GMM Diagonal T2 - Diagonal Tied NT16"), (GMMclassify, "GMM Diagonal T2 - Diagonal Tied NT32"),
                      (GMMclassify, "GMM Diagonal T4 - Full NT1)"), (GMMclassify, "GMM Diagonal T4 - Full NT2"), (GMMclassify, "GMM Diagonal T4 - Full NT4"),
                      (GMMclassify, "GMM Diagonal T4 - Full NT8)"), (GMMclassify, "GMM Diagonal T4 - Full NT16"), (GMMclassify, "GMM Diagonal T4 - Full NT32"),
                      (GMMclassify, "GMM Diagonal T4 - Diagonal Tied NT1)"), (GMMclassify, "GMM Diagonal T4 - Diagonal Tied NT2"), (GMMclassify, "GMM Diagonal T4 - Diagonal Tied NT4"),
                      (GMMclassify, "GMM Diagonal T4 - Diagonal Tied NT8)"), (GMMclassify, "GMM Diagonal T4 - Diagonal Tied NT16"), (GMMclassify, "GMM Diagonal T4 - Diagonal Tied NT32")
                    ]

    '''(GMMclassify, "GMM Full (T2 - NT1)"), (GMMclassify, "GMM Full (T2 - NT2)"), (GMMclassify, "GMM Full (T2 - NT4)"),
                      (GMMclassify, "GMM Full (T2 - NT8)"), (GMMclassify, "GMM Full (T2 - NT16)"), (GMMclassify, "GMM Full (T2- NT32)"),
                      (GMMclassify, "GMM Full (T4 - NT1)"), (GMMclassify, "GMM Full (T4 - NT2)"), (GMMclassify, "GMM Full (T4 - NT4)"),
                      (GMMclassify, "GMM Full (T4 - NT8)"), (GMMclassify, "GMM Full (T4 - NT16)"), (GMMclassify, "GMM Full (T4 - NT32)"),
                      (GMMclassify, "GMM Diagonal (T1 - NT1)"), (GMMclassify, "GMM Diagonal (T1 - NT2)"), (GMMclassify, "GMM Diagonal (T1 - NT4)"),
                      (GMMclassify, "GMM Diagonal (T1 - NT8)"), (GMMclassify, "GMM Diagonal (T1 - NT16)"), (GMMclassify, "GMM Diagonal (T1 - NT32)"),
                      (GMMclassify, "GMM Diagonal (T2 - NT1)"), (GMMclassify, "GMM Diagonal (T2 - NT2)"), (GMMclassify, "GMM Diagonal (T2 - NT4)"),
                      (GMMclassify, "GMM Diagonal (T2 - NT8)"), (GMMclassify, "GMM Diagonal (T2 - NT16)"), (GMMclassify, "GMM Diagonal (T2 - NT32)"),
                      (GMMclassify, "GMM Diagonal (T4 - NT1)"), (GMMclassify, "GMM Diagonal (T4 - NT2)"), (GMMclassify, "GMM Diagonal (T4 - NT4)"),
                      (GMMclassify, "GMM Diagonal (T4 - NT8)"), (GMMclassify, "GMM Diagonal (T4 - NT16)"), (GMMclassify, "GMM Diagonal (T4 - NT32)"),
                      (GMMclassify, "GMM Tied (T1 - NT1)"), (GMMclassify, "GMM Tied (T1 - NT2)"), (GMMclassify, "GMM Tied (T1 - NT4)"),
                      (GMMclassify, "GMM Tied (T1 - NT8)"), (GMMclassify, "GMM Tied (T1 - NT16)"), (GMMclassify, "GMM Tied (T1 - NT32)"),
                      (GMMclassify, "GMM Tied (T2 - NT1)"), (GMMclassify, "GMM Tied (T2 - NT2)"), (GMMclassify, "GMM Tied (T2 - NT4)"),
                      (GMMclassify, "GMM Tied (T2 - NT8)"), (GMMclassify, "GMM Tied (T2 - NT16)"), (GMMclassify, "GMM Tied (T2 - NT32)"),
                      (GMMclassify, "GMM Tied (T4 - NT1)"), (GMMclassify, "GMM Tied (T4 - NT2)"), (GMMclassify, "GMM Tied (T4 - NT4)"),
                      (GMMclassify, "GMM Tied (T4 - NT8)"), (GMMclassify, "GMM Tied (T4 - NT16)"), (GMMclassify, "GMM Tied (T4 - NT32)")]'''

    # vanno modificati nei parametri (distinguiamo il numero di componenti NT - T e le varianti NT - T)
    parametersGMM = [(("Diagonal", "Default"), (1, 1), 0.01, 0.1), (("Diagonal", "Default"), (2, 1), 0.01, 0.1), (("Diagonal", "Default"), (4, 1), 0.01, 0.1),
                     (("Diagonal", "Default"), (8, 1), 0.01, 0.1), (("Diagonal", "Default"), (16, 1), 0.01, 0.1), (("Diagonal", "Default"), (32, 1), 0.01, 0.1),
                     (("Diagonal tied", "Default"), (1, 1), 0.01, 0.1), (("Diagonal tied", "Default"), (2, 1), 0.01, 0.1), (("Diagonal tied", "Default"), (4, 1), 0.01, 0.1),
                     (("Diagonal tied", "Default"), (8, 1), 0.01, 0.1), (("Diagonal tied", "Default"), (16, 1), 0.01, 0.1), (("Diagonal tied", "Default"), (32, 1), 0.01, 0.1),
                     (("Diagonal", "Default"), (1, 2), 0.01, 0.1), (("Diagonal", "Default"), (2, 2), 0.01, 0.1), (("Diagonal", "Default"), (4, 2), 0.01, 0.1),
                     (("Diagonal", "Default"), (8, 2), 0.01, 0.1), (("Diagonal", "Default"), (16, 2), 0.01, 0.1), (("Diagonal", "Default"), (32, 2), 0.01, 0.1),
                     (("Diagonal tied", "Default"), (1, 2), 0.01, 0.1), (("Diagonal tied", "Default"), (2, 2), 0.01, 0.1), (("Diagonal tied", "Default"), (4, 2), 0.01, 0.1),
                     (("Diagonal tied", "Default"), (8, 2), 0.01, 0.1), (("Diagonal tied", "Default"), (16, 2), 0.01, 0.1), (("Diagonal tied", "Default"), (32, 2), 0.01, 0.1),
                     (("Diagonal", "Default"), (1, 4), 0.01, 0.1), (("Diagonal", "Default"), (2, 4), 0.01, 0.1), (("Diagonal", "Default"), (4, 4), 0.01, 0.1),
                     (("Diagonal", "Default"), (8, 4), 0.01, 0.1), (("Diagonal", "Default"), (16, 4), 0.01, 0.1), (("Diagonal", "Default"), (32, 4), 0.01, 0.1),
                     (("Diagonal tied", "Default"), (1, 4), 0.01, 0.1), (("Diagonal tied", "Default"), (2, 4), 0.01, 0.1), (("Diagonal tied", "Default"), (4, 4), 0.01, 0.1),
                     (("Diagonal tied", "Default"), (8, 4), 0.01, 0.1), (("Diagonal tied", "Default"), (16, 4), 0.01, 0.1), (("Diagonal tied", "Default"), (32, 4), 0.01, 0.1),

                     (("Default", "Diagonal"), (1, 1), 0.01, 0.1), (("Default", "Diagonal"), (2, 1), 0.01, 0.1), (("Default", "Diagonal"), (4, 1), 0.01, 0.1),
                     (("Default", "Diagonal"), (8, 1), 0.01, 0.1), (("Default", "Diagonal"), (16, 1), 0.01, 0.1), (("Default", "Diagonal"), (32, 1), 0.01, 0.1),
                     (("Diagonal tied", "Diagonal"), (1, 1), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (2, 1), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (4, 1), 0.01, 0.1),
                     (("Diagonal tied", "Diagonal"), (8, 1), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (16, 1), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (32, 1), 0.01, 0.1),
                     (("Default", "Diagonal"), (1, 2), 0.01, 0.1), (("Diagonal", "Diagonal"), (2, 2), 0.01, 0.1), (("Default", "Diagonal"), (4, 2), 0.01, 0.1),
                     (("Default", "Diagonal"), (8, 2), 0.01, 0.1), (("Diagonal", "Diagonal"), (16, 2), 0.01, 0.1), (("Default", "Diagonal"), (32, 2), 0.01, 0.1),
                     (("Diagonal tied", "Diagonal"), (1, 2), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (2, 2), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (4, 2), 0.01, 0.1),
                     (("Diagonal tied", "Diagonal"), (8, 2), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (16, 2), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (32, 2), 0.01, 0.1),
                     (("Default", "Diagonal"), (1, 4), 0.01, 0.1), (("Default", "Diagonal"), (2, 4), 0.01, 0.1), (("Default", "Diagonal"), (4, 4), 0.01, 0.1),
                     (("Default", "Diagonal"), (8, 4), 0.01, 0.1), (("Default", "Diagonal"), (16, 4), 0.01, 0.1), (("Default", "Diagonal"), (32, 4), 0.01, 0.1),
                     (("Diagonal tied", "Diagonal"), (1, 4), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (2, 4), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (4, 4), 0.01, 0.1),
                     (("Diagonal tied", "Diagonal"), (8, 4), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (16, 4), 0.01, 0.1), (("Diagonal tied", "Diagonal"), (32, 4), 0.01, 0.1)
                     ]


    '''("Default", (1, 4), 0.01, 0.1), ("Default", (2, 4), 0.01, 0.1), ("Default", (4, 4), 0.01, 0.1),
                     ("Default", (8, 4), 0.01, 0.1), ("Default", (16, 4), 0.01, 0.1), ("Default", (32, 4), 0.01, 0.1),
                     ("Diagonal", (1, 1), 0.01, 0.1), ("Diagonal", (2, 1), 0.01, 0.1), ("Diagonal", (4, 1), 0.01, 0.1),
                     ("Diagonal", (8, 1), 0.01, 0.1), ("Diagonal", (16, 1), 0.01, 0.1), ("Diagonal", (32, 1), 0.01, 0.1),
                     ("Diagonal", (1, 2), 0.01, 0.1), ("Diagonal", (2, 2), 0.01, 0.1), ("Diagonal", (4, 2), 0.01, 0.1),
                     ("Diagonal", (8, 2), 0.01, 0.1), ("Diagonal", (16, 2), 0.01, 0.1), ("Diagonal", (32, 2), 0.01, 0.1),
                     ("Diagonal", (1, 4), 0.01, 0.1), ("Diagonal", (2, 4), 0.01, 0.1), ("Diagonal", (4, 4), 0.01, 0.1),
                     ("Diagonal", (8, 4), 0.01, 0.1), ("Diagonal", (16, 4), 0.01, 0.1), ("Diagonal", (32, 4), 0.01, 0.1),
                     ("Tied", (1, 1), 0.01, 0.1), ("Tied", (2, 1), 0.01, 0.1), ("Tied", (4, 1), 0.01, 0.1),
                     ("Tied", (8, 1), 0.01, 0.1), ("Tied", (16, 1), 0.01, 0.1), ("Tied", (32, 1), 0.01, 0.1),
                     ("Tied", (1, 2), 0.01, 0.1), ("Tied", (2, 2), 0.01, 0.1), ("Tied", (4, 2), 0.01, 0.1),
                     ("Tied", (8, 2), 0.01, 0.1), ("Tied", (16, 2), 0.01, 0.1), ("Tied", (32, 2), 0.01, 0.1),
                     ("Tied", (1, 4), 0.01, 0.1), ("Tied", (2, 4), 0.01, 0.1), ("Tied", (4, 4), 0.01, 0.1),
                     ("Tied", (8, 4), 0.01, 0.1), ("Tied", (16, 4), 0.01, 0.1), ("Tied", (32, 4), 0.01, 0.1)]'''

    classifiersMVG = [(MVG_log, "Multivariate Gaussian Classifier"), (MVG_log, "Naive Bayes Gaussian"),
                      (MVG_log, "Tied Covariance Gaussian")]
    parametersMVG = [("Default"), ("Naive"), ("Tied")]

    # show_histo(dataset_train, labels_train)
    # show_histo(zNormalization(dataset_train), labels_train)

    # heatmap(dataset_train[:, labels_train == 0], "Blues")
    # heatmap(dataset_train[:, labels_train == 1], "Reds")
    # heatmap(dataset_train, "Greys")
    minDCF = numpy.zeros(0)

    # ----- MVG -------
    '''print(":------ MVG classifiers - NO PCA - 5 fold (0.5, 1, 1) ------:")
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
    print("PCA 4  - Tied: %f" % ((minDCF[11]+minDCF[23])/2))'''

    # ----- LR -------
    '''workingPoint = (0.5, 1, 1)
    print(":----- LR classifiers - NO PCA - 5 fold (0.5, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, classifiersLR, parametersLR))
    print(":----- LR classifiers -  PCA 5 - 5 fold (0.5, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, classifiersLR, parametersLR))
    print(":----- LR classifiers - Z norm - 5 fold (0.5, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiersLR, parametersLR))
    print(":----- LR classifiers -  PCA 4 - 5 fold (0.5, 1, 1) --------:")
    dataset = PCA(dataset_train, labels_train, 4)
    minDCF = numpy.append(minDCF, kfold(dataset, labels_train, k, workingPoint, classifiersLR, parametersLR))

    workingPoint = (0.1, 1, 1)
    print(":----- LR classifiers - NO PCA - 5 fold (0.1, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, classifiersLR, parametersLR))
    print(":----- LR classifiers -  PCA 5 - 5 fold (0.1, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, classifiersLR, parametersLR))
    print(":----- LR classifiers - Z norm - 5 fold (0.1, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiersLR, parametersLR))
    print(":----- LR classifiers -  PCA 4 - 5 fold (0.1, 1, 1) --------:")
    dataset = PCA(dataset_train, labels_train, 4)
    minDCF = numpy.append(minDCF, kfold(dataset, labels_train, k, workingPoint, classifiersLR, parametersLR))

    print(":----------------------- Average minDCF --------------------:")
    print("NO PCA -   LR: %f" % ((minDCF[0]+minDCF[12])/2))
    print("NO PCA -  WLR: %f" % ((minDCF[1]+minDCF[13])/2))
    print("NO PCA -  QLR: %f" % ((minDCF[2]+minDCF[14])/2))
    print("NO PCA - WQLR: %f" % ((minDCF[3]+minDCF[15])/2))
    print("PCA  5 -   LR: %f" % ((minDCF[4]+minDCF[16])/2))
    print("PCA  5 -  WLR: %f" % ((minDCF[5]+minDCF[17])/2))
    print("PCA  5 -  QLR: %f" % ((minDCF[6]+minDCF[18])/2))
    print("PCA  5 - WQLR: %f" % ((minDCF[7]+minDCF[19])/2))
    print("Z NORM -   LR: %f" % ((minDCF[8]+minDCF[20])/2))
    print("Z NORM -  WLR: %f" % ((minDCF[9]+minDCF[21])/2))
    print("Z NORM -  QLR: %f" % ((minDCF[10]+minDCF[22])/2))
    print("Z NORM - WQLR: %f" % ((minDCF[11]+minDCF[23])/2))'''

    # ---- SVM ---- (poly solo con c=1)
    '''K = 1.0; c = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    for C in c:
        print("---------------------------- C = %f -------------------------" % (c))
        minDCF = numpy.zeros(0)
        workingPoint = (0.5, 1, 1)
        print(":---- Linear SVM - NO PCA - 5 fold (0.5, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)]))
        print(":---- Linear SVM - PCA  5 - 5 fold (0.5, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)]))
        print(":---- Linear SVM - Z NORM - 5 fold (0.5, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint, [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)]))

        workingPoint = (0.1, 1, 1)
        print(":---- Linear SVM - NO PCA - 5 fold (0.1, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)]))
        print(":---- Linear SVM - PCA  5 - 5 fold (0.1, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)]))
        print(":---- Linear SVM - Z NORM - 5 fold (0.1, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint, [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)]))

        print(":----------------------- Average minDCF --------------------:")
        print("NO PCA - Linear: %f" % ((minDCF[0]+minDCF[3])/2))
        print("PCA  5 - Linear: %f" % ((minDCF[1]+minDCF[4])/2))
        print("Z NORM - Linear: %f" % ((minDCF[2]+minDCF[5])/2)) 
    '''
    # SVM BEST
    '''K = 1.0; d = 2; gamma = 0.001; p = 0.5
    c = [0.001, 0.01, 0.1, 1, 10, 100]
    for C in c:
        print("------------------------ C = %f -----------------------------" % (C))
        minDCF = numpy.zeros(0)
        workingPoint = (0.5, 1, 1)
        print(":---- SVM Poly 2/RBF - PCA  5 - 5 fold (0.5, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")],
                                            [("SVMRBF", K, C, gamma, p)]))
        minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelPoly, "Support Vector Machine - Poly")],
                                            [("SVMP", K, C, 1, d, p)]))
        workingPoint = (0.1, 1, 1)
        print(":---- SVM Poly 2/RBF - PCA  5 - 5 fold (0.1, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")],
                                            [("SVMRBF", K, C, gamma, p)]))
        minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelPoly, "Support Vector Machine - Poly")],
                                            [("SVMP", K, C, 1, d, p)]))

        print(":----------------------- Average minDCF --------------------:")
        print("PCA 5 - Poly: %f" % ((minDCF[0]+minDCF[2])/2))
        print("PCA 5 -  RBF: %f\n" % ((minDCF[1]+minDCF[3])/2))
    '''
    '''
    K = 1.0; d = 3
    c = [0.001, 0.01, 0.1, 1, 10, 100]; Gamma = [0.001, 0.0001, 0.00001]; gamma = 0.00001
    for C in c:
        print("------------------------ C = %f -----------------------------" % (C))
        minDCF = numpy.zeros(0)
        workingPoint = (0.5, 1, 1)
        print(":---- Linear SVM - NO PCA - 5 fold (0.5, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")],
                                            [("SVMRBF", K, C, gamma)]))
        print(":---- Linear SVM - PCA  5 - 5 fold (0.5, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")],
                                            [("SVMRBF", K, C, gamma)]))
        print(":---- Linear SVM - Z NORM - 5 fold (0.5, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")],
                                            [("SVMRBF", K, C, gamma)]))

        workingPoint = (0.1, 1, 1)
        print(":---- Linear SVM - NO PCA - 5 fold (0.1, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")],
                                            [("SVMRBF", K, C, gamma)]))
        print(":---- Linear SVM - PCA  5 - 5 fold (0.1, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")],
                                            [("SVMRBF", K, C, gamma)]))
        print(":---- Linear SVM - Z NORM - 5 fold (0.1, 1, 1) --------------")
        minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint,
                                            [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")],
                                            [("SVMRBF", K, C, gamma)]))

        print(":----------------------- Average minDCF --------------------:")
        print("NO PCA - Linear: %f" % ((minDCF[0] + minDCF[3]) / 2))
        print("PCA  5 - Linear: %f" % ((minDCF[1] + minDCF[4]) / 2))
        print("Z NORM - Linear: %f\n" % ((minDCF[2] + minDCF[5]) / 2))
    '''

    '''print(0)
    kfoldPlotMinDCFC(dataset_train, labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)])
    print(0)
    kfoldPlotMinDCFC(zNormalization(dataset_train), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)])
    print(0)
    kfoldPlotMinDCFC(PCA(dataset_train, labels_train, 5), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)])
    print(0)
    kfoldPlotMinDCFC(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(SupportVectorMachineLinear, "Support Vector Machine - Linear")], [("SVML", K, C)])'''


    '''print(0)
    kfoldPlotMinDCFC(dataset_train, labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelPoly, "Support Vector Machine - Poly")], [("SVMP", K, C, c, d)])
    print(0)
    kfoldPlotMinDCFC(zNormalization(dataset_train), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelPoly, "Support Vector Machine - Poly")], [("SVMP", K, C, c, d)])
    print(0)
    kfoldPlotMinDCFC(PCA(dataset_train, labels_train, 5), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelPoly, "Support Vector Machine - Poly")], [("SVMP", K, C, c, d)])
    print(0)
    kfoldPlotMinDCFC(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelPoly, "Support Vector Machine - Poly")], [("SVMP", K, C, c, d)])'''

    '''print(0)
    kfoldPlotMinDCFC(dataset_train, labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelPoly, "Support Vector Machine - Poly")], [("SVMP", 1.0, C, 1, 3)])
    print(0)
    kfoldPlotMinDCFC(PCA(dataset_train, labels_train, 5), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelPoly, "Support Vector Machine - Poly")], [("SVMP", 1.0, C, 1, 3)])'''

    '''print(0)
    kfoldPlotMinDCFC(zNormalization(dataset_train), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", K, C, 0.001)])
    print(0)
    kfoldPlotMinDCFC(zNormalization(dataset_train), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", K, C, 0.0001)])
    print(0)
    kfoldPlotMinDCFC(zNormalization(dataset_train), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", K, C, 0.00001)])'''

    '''print(0)
    kfoldPlotMinDCFC(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", K, C, 0.001)])
    print(0)
    kfoldPlotMinDCFC(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", K, C, 0.0001)])
    print(0)
    kfoldPlotMinDCFC(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)],
                     [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", K, C, 0.00001)])

    plt.xlabel("C")
    plt.ylabel("minCprim")
    plt.xscale('log')
    plt.xlim([10**-5, 10**2])
    plt.grid(True)
    plt.legend(["SVM RBF (z-norm) (PCA 5) (γ=10e-3)", "SVM RBF (z-norm) (PCA 5) (γ=10e-4)", "SVM RBF (z-norm) (PCA 5) (γ=10e-5)"])
    plt.savefig('SVM-RBF(PCA5z-norm).svg')
    plt.show()'''

    # -------------- GMM ------------------
    '''workingPoint = (0.5, 1, 1)
    print(":---- GMM Classifiers - PCA 5  - 5 fold (0.5, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, workingPoint, classifiersGMM, parametersGMM))
    print(":---- GMM Classifiers - PCA 5  - 5 fold (0.1, 1, 1) --------:")
    workingPoint = (0.1, 1, 1)
    minDCF = numpy.append(minDCF, kfold(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, workingPoint, classifiersGMM, parametersGMM))

    print(":------------------- Average minDCF ------------------------:")
    for i in range(len(classifiersGMM)):
        print("%s: %f" % (classifiersGMM[i][1], (minDCF[i]+minDCF[len(classifiersGMM)+i])/2))'''


    # PROVE DA QUI
    '''print(":----- LR classifiers - NO PCA - 5 fold (0.5, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, classifiersLR, parametersLR0))
    print(":--- LR classifiers - NO PCA - Znorm - 5 fold (0.5, 1, 1) --:")
    minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiersLR, parametersLR0))

    workingPoint = (0.1, 1, 1)
    print(":----- LR classifiers - NO PCA - 5 fold (0.1, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, classifiersLR, parametersLR0))
    print(":--- LR classifiers - NO PCA - Znorm - 5 fold (0.1, 1, 1) --:")
    minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiersLR, parametersLR0))
    minDCF = numpy.zeros(0)


    print(":----- LR classifiers - NO PCA - 5 fold (0.5, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiersLR, parametersLR))

    workingPoint = (0.1, 1, 1)
    print(":----- LR classifiers - NO PCA - 5 fold (0.1, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiersLR, parametersLR))

    print(":----------------------- Average minDCF --------------------:")
    print("NO PCA -   LR: %f" % ((minDCF[0] + minDCF[4]) / 2))
    print("NO PCA -  WLR: %f" % ((minDCF[1] + minDCF[5]) / 2))
    print("NO PCA -  QLR: %f" % ((minDCF[2] + minDCF[6]) / 2))
    print("NO PCA - WQLR: %f" % ((minDCF[3] + minDCF[7]) / 2))'''

    '''print(":----- LR classifiers - NO PCA - 5 fold (0.5, 1, 1) --------:")
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, [(logisticRegression, "Logistic Regression")], [("Default", 0.00001)]))
    workingPoint = (0.1, 1, 1)
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, [(logisticRegression, "Logistic Regression")], [("Default", 0.00001)]))'''

    '''kfoldPlotMinDCFlambda(dataset_train, labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(logisticRegression, "Quadratic Logistic Regression")], [("Quadratic", 0.001)])
    print("1")
    kfoldPlotMinDCFlambda(zNormalization(dataset_train), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(logisticRegression, "Quadratic Logistic Regression")], [("Quadratic", 0.001)])
    print("2")
    kfoldPlotMinDCFlambda(PCA(dataset_train, labels_train, 5), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(logisticRegression, "Quadratic Logistic Regression")], [("Quadratic", 0.001)])
    print("3")
    kfoldPlotMinDCFlambda(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(logisticRegression, "Quadratic Logistic Regression")], [("Quadratic", 0.001)])
    print("4")
    kfoldPlotMinDCFlambda(PCA(dataset_train, labels_train, 4), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(logisticRegression, "Quadratic Logistic Regression")], [("Quadratic", 0.001)])
    print("5")
    kfoldPlotMinDCFlambda(zNormalization(PCA(dataset_train, labels_train, 4)), labels_train, k, [(0.5, 1, 1), (0.1, 1, 1)], [(logisticRegression, "Quadratic Logistic Regression")], [("Quadratic", 0.001)])

    plt.xlabel("λ")
    plt.ylabel("minCprim")
    plt.xscale('log')
    plt.xlim([10**-5, 10**3])
    plt.grid(True)
    plt.legend(["Quad Log-Reg", "Quad Log-Reg (z-norm)", "Quad Log-Reg (PCA 5)", "Quad Log-Reg (z-norm) (PCA 5)", "Quad Log-Reg (PCA 4)", "Quad Log-Reg (z-norm) (PCA 4)"])
    plt.savefig('Quad-Log-Reg.svg')
    plt.show()'''

    # -------- DET -----------
    '''parametersDET = [(("Default", "Default"), (32, 4), 0.01, 0.1), ("SVMRBF", 1, 10, 0.001), ("Quadratic", 10)]
    classifiersDET = [(GMMclassify, "GMM Full T4 - NT 32"), (SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF"), (logisticRegression, "Quadratic Logistic Regression")]

    print(0)
    kfold(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, workingPoint, [(GMMclassify, "GMM Full T4 - NT 32")], [(("Default", "Default"), (32, 4), 0.01, 0.1)], True)
    print(0)
    kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")], [("SVMRBF", 1, 10, 0.001)], True)
    print(0)
    kfold(dataset_train, labels_train, k, workingPoint, [(logisticRegression, "Quadratic Logistic Regression")], [("Quadratic", 10)], True)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend([classifiersDET[0][1], classifiersDET[1][1], classifiersDET[2][1]])
    plt.show()'''

    # Score calibration (prova su SVM - score calibration mediante Prior weighted Log-Reg con lambda=10e-4)
    '''print(0)
    kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", 1, 10, 0.001, 0.5)])
    print(0)
    kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", 1, 10, 0.001, 0.5)], toCalibrate=True)'''

    ''' # FUNZIONA ! 
    print(0)
    kfoldBayesErrorPlot(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, SupportVectorMachineKernelRBF, ("SVMRBF", 1, 10, 0.001, 0.5))
    print(0)
    kfoldBayesErrorPlot(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, SupportVectorMachineKernelRBF, ("SVMRBF", 1, 10, 0.001, 0.5), toCalibrate=True)
    plt.xlabel("prior log-odds")
    plt.ylabel("Cprim")
    plt.legend(["RBF (DCF) ", "RBF (minDCF)", "RBF-cal (DCF)", "RBF-cal (minDCF)"])
    plt.savefig("RBF-bayesErrorPlot(K=1,C=10,gamma=10e-3).svg")
    plt.show()
    '''

    # kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(logisticRegression, "Logistic Regression Weighted")], [("Weighted", 0.0001, 0.5)], toCalibrate=False)
    # Model fusion
    '''print(0)
    kfoldBayesErrorPlot(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, SupportVectorMachineKernelRBF, ("SVMRBF", 1, 10, 0.001, 0.5), toCalibrate=True)
    print(1)
    kfoldBayesErrorPlot(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, GMMclassify, (("Diagonal tied", "Default"), (32, 4), 0.01, 0.1))
    print(2)
    kfoldFusion(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineKernelRBF, "SVM - RBF"), (GMMclassify, "GMM")], [("SVMRBF", 1, 10, 0.001, 0.5), (("Diagonal tied", "Default"), (32, 4), 0.01, 0.1)], [True, False])

    plt.xlabel("prior log-odds")
    plt.ylabel("Cprim")
    plt.legend(["RBF (DCF) ", "RBF (minDCF)", "RBF-cal (DCF)", "RBF-cal (minDCF)", "Fusion (DCF)", "Fusion (minDCF)"])
    plt.savefig("RBF-bayesErrorPlot(K=1,C=10,gamma=10e-3).svg")
    plt.show()'''

    # ---- Final Fusion Training set ----
    '''print(0)
    kfoldBayesErrorPlot(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, logisticRegression, ("Weighted quadratic", 10, 0.1), color="r", toCalibrate=False)
    print(1)
    kfoldBayesErrorPlot(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, SupportVectorMachineKernelRBF, ("SVMRBF", 1, 10, 0.001, 0.5), color="b", toCalibrate=False)
    print(2)
    kfoldBayesErrorPlot(dataset_train, labels_train, k, workingPoint, GMMclassify, (("Tied", "Tied"), (32, 2), 0.01, 0.1), color="g", toCalibrate=False)
    print(3)
    kfoldFusion(dataset_train, labels_train, k, workingPoint, [(logisticRegression, "Weighted quadratic log-reg"), (SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF"), (GMMclassify, "GMM")], [("Weighted quadratic", 10, 0.1), ("SVMRBF", 1, 10, 0.001, 0.5), (("Tied", "Tied"), (32, 2), 0.01, 0.1)], [False, False, False], pca=[5, 5, -1])

    plt.xlabel("prior log-odds")
    plt.ylabel("Cprim")
    plt.legend(["W. Quad Log-Reg (actDCF)", "W. Quad Log-Reg (minDCF)", "SVM RBF (actDCF)", "SVM RBF (minDCF)", "GMM (actDCF)", "GMM (minDCF)", "Fusion (actDCF)", "Fusion (minDCF)"])
    plt.savefig("BayesErrorPlot(FinalCalibrated2).svg")
    plt.show()'''

    # ---- Final Fusion Test Set ----



    # ---- Final Bayes Error Plot - raw scores & calibration ----
    '''workingPoint = (0.5, 1, 1)
    print(0)
    kfoldBayesErrorPlot(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, logisticRegression, ("Weighted quadratic", 0.1), color="r", toCalibrate=True)
    print(1)
    kfoldBayesErrorPlot(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, SupportVectorMachineKernelRBF, ("SVMRBF", 1, 10, 0.001, 0.5), color="b", toCalibrate=True)
    print(2)
    kfoldBayesErrorPlot(zNormalization(PCA(dataset_train, labels_train, 5)), labels_train, k, workingPoint, GMMclassify, (("Default", "Default"), (32, 4), 0.01, 0.1), color="g", toCalibrate=True)

    plt.xlabel("prior log-odds")
    plt.ylabel("Cprim")
    plt.legend(["W. Quad Log-Reg (actDCF)", "W. Quad Log-Reg (minDCF)", "SVM RBF (actDCF)", "SVM RBF (minDCF)", "GMM (actDCF)", "GMM (minDCF)"])
    plt.savefig("BayesErrorPlot(FinalCalibrated).svg")
    plt.show()'''

    # Results after the whole testing of the best models
    '''print("------------ MinDCF Average - Raw scores -------------")
    workingPoint = (0.5, 1, 1)
    minDCF = numpy.zeros(0)
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(GMMclassify, "GMM Tied (T2 - NT32)")], [(("Tied", "Tied"), (32, 2), 0.01, 0.1)]))
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", 1, 10, 0.001, 0.5)]))
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, [(logisticRegression, "Weighted Quadratic Logistic Regression")], [("Weighted quadratic", 10, 0.1)]))
    workingPoint = (0.1, 1, 1)
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(GMMclassify, "GMM Tied (T2 - NT32)")], [(("Tied", "Tied"), (32, 2), 0.01, 0.1)]))
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", 1, 10, 0.001, 0.5)]))
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, [(logisticRegression, "Weighted Quadratic Logistic Regression")], [("Weighted quadratic", 10, 0.1)]))
    print("\nGMM: %f" % ((minDCF[0]+minDCF[3])/2))
    print("RBF: %f" % ((minDCF[1]+minDCF[4])/2))
    print("WQLR: %f\n" % ((minDCF[2]+minDCF[5])/2))

    print("----------- MinDCF Average - Calibrated --------------")
    minDCF = numpy.zeros(0)
    workingPoint = (0.5, 1, 1)
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(GMMclassify, "GMM Tied (T2 - NT32)")], [(("Tied", "Tied"), (32, 2), 0.01, 0.1)], toCalibrate=True))
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", 1, 10, 0.001, 0.5)], toCalibrate=True))
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, [(logisticRegression, "Weighted Quadratic Logistic Regression")], [("Weighted quadratic", 10, 0.1)], toCalibrate=True))
    workingPoint = (0.1, 1, 1)
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(GMMclassify, "GMM Tied (T2 - NT32)")], [(("Tied", "Tied"), (32, 2), 0.01, 0.1)], toCalibrate=True))
    minDCF = numpy.append(minDCF, kfold(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoint, [(SupportVectorMachineKernelRBF, "Support Vector Machine - RBF")], [("SVMRBF", 1, 10, 0.001, 0.5)], toCalibrate=True))
    minDCF = numpy.append(minDCF, kfold(dataset_train, labels_train, k, workingPoint, [(logisticRegression, "Weighted Quadratic Logistic Regression")], [("Weighted quadratic", 10, 0.1)], toCalibrate=True))
    print("\nGMM: %f" % ((minDCF[0]+minDCF[3])/2))
    print("RBF: %f" % ((minDCF[1]+minDCF[4])/2))
    print("WQLR: %f\n" % ((minDCF[2]+minDCF[5])/2))'''


    # ----- EVALUATION ON THE THREE CANDIDATES -----
    '''print("--------------- MinDCF Average - Raw scores -----------------")
    workingPoint = (0.5, 1, 1)
    P = PCA(dataset_train, labels_train, 5, eval=True)
    # muPCA, varPCA = zNormalization(PCA(dataset_train, labels_train, 5), eval=True)

    scoreseval = GMMclassify(PCA(dataset_train, labels_train, 5), labels_train, numpy.dot(P.transpose(), dataset_test), labels_test, 0.5, (("Tied", "Tied"), (32, 2), 0.01, 0.1))
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF1 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF1 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    workingPoint = (0.1, 1, 1)
    scoreseval = GMMclassify(PCA(dataset_train, labels_train, 5), labels_train, numpy.dot(P.transpose(), dataset_test), labels_test, 0.1, (("Tied", "Tied"), (32, 2), 0.01, 0.1))
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF2 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF2 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    print(f"GMM results average:\nActualDCF: {(actualDCF1+actualDCF2)/2}\nMinDCF: {(minDCF1+minDCF2)/2}\n")

    workingPoint = (0.5, 1, 1)
    P = PCA(dataset_train, labels_train, 5, eval=True)

    scoreseval = SupportVectorMachineKernelRBF((PCA(dataset_train, labels_train, 5)), labels_train, (numpy.dot(P.transpose(), dataset_test)), labels_test, 0.5, ("SVMRBF", 1, 10, 0.001, 0.5), scoresFlag=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF1 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF1 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    workingPoint = (0.1, 1, 1)
    scoreseval = SupportVectorMachineKernelRBF((PCA(dataset_train, labels_train, 5)), labels_train, (numpy.dot(P.transpose(), dataset_test)), labels_test, 0.1, ("SVMRBF", 1, 10, 0.001, 0.5), scoresFlag=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF2 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF2 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    print(f"SVM RBF results average:\nActualDCF: {(actualDCF1+actualDCF2)/2}\nMinDCF: {(minDCF1+minDCF2)/2}\n")

    workingPoint = (0.5, 1, 1)
    P = PCA(dataset_train, labels_train, 5, eval=True)

    scoreseval = logisticRegression((PCA(dataset_train, labels_train, 5)), labels_train, (numpy.dot(P.transpose(), dataset_test)), labels_test, 0.5, ("Weighted quadratic", 10, 0.1), score=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF1 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF1 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    workingPoint = (0.1, 1, 1)
    scoreseval = logisticRegression((PCA(dataset_train, labels_train, 5)), labels_train, (numpy.dot(P.transpose(), dataset_test)), labels_test, 0.1, ("Weighted quadratic", 10, 0.1), score=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF2 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF2 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    print(f"WQLR results average:\nActualDCF: {(actualDCF1+actualDCF2)/2}\nMinDCF: {(minDCF1+minDCF2)/2}\n")

    print("--------------- MinDCF Average - Calibrated -----------------")
    workingPoint = (0.5, 1, 1)
    P = PCA(dataset_train, labels_train, 5, eval=True)
    # muPCA, varPCA = zNormalization(PCA(dataset_train, labels_train, 5), eval=True)

    scoreseval = GMMclassify(PCA(dataset_train, labels_train, 5), labels_train, numpy.dot(P.transpose(), dataset_test), labels_test, 0.5, (("Tied", "Tied"), (32, 2), 0.01, 0.1), toCalibrate=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF1 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF1 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    workingPoint = (0.1, 1, 1)
    scoreseval = GMMclassify(PCA(dataset_train, labels_train, 5), labels_train, numpy.dot(P.transpose(), dataset_test), labels_test, 0.1, (("Tied", "Tied"), (32, 2), 0.01, 0.1), toCalibrate=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF2 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF2 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    print(f"GMM results average:\nActualDCF: {(actualDCF1+actualDCF2)/2}\nMinDCF: {(minDCF1+minDCF2)/2}\n")

    workingPoint = (0.5, 1, 1)
    P = PCA(dataset_train, labels_train, 5, eval=True)

    scoreseval = SupportVectorMachineKernelRBF((PCA(dataset_train, labels_train, 5)), labels_train, (numpy.dot(P.transpose(), dataset_test)), labels_test, 0.5, ("SVMRBF", 1, 10, 0.001), scoresFlag=True, toCalibrate=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF1 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF1 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    workingPoint = (0.1, 1, 1)
    scoreseval = SupportVectorMachineKernelRBF((PCA(dataset_train, labels_train, 5)), labels_train, (numpy.dot(P.transpose(), dataset_test)), labels_test, 0.1, ("SVMRBF", 1, 10, 0.001), scoresFlag=True, toCalibrate=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF2 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF2 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    print(f"SVM RBF results average:\nActualDCF: {(actualDCF1+actualDCF2)/2}\nMinDCF: {(minDCF1+minDCF2)/2}\n")

    workingPoint = (0.5, 1, 1)
    P = PCA(dataset_train, labels_train, 5, eval=True)

    scoreseval = logisticRegression(dataset_train, labels_train, dataset_test, labels_test, 0.5, ("Weighted quadratic", 10, 0.1), score=True, toCalibrate=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF1 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF1 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    workingPoint = (0.1, 1, 1)
    scoreseval = logisticRegression(dataset_train, labels_train, dataset_test, labels_test, 0.1, ("Weighted quadratic", 10, 0.1), score=True, toCalibrate=True)
    cm = optimal_bayes_decisions(scoreseval, labels_test, workingPoint)
    DCFu = compute_bayes_risk(cm, workingPoint)
    actualDCF2 = DCFu / compute_dummy_bayes(workingPoint)
    minDCF2 = compute_minDCF(scoreseval, labels_test, workingPoint, True)

    print(f"WQLR results average:\nActualDCF: {(actualDCF1+actualDCF2)/2}\nMinDCF: {(minDCF1+minDCF2)/2}\n")'''

    # ----- Fusion for the test set -----
    '''effPriorLogOdds = numpy.linspace(-3, 3, 21)
    pi_sign = 1/(1+numpy.exp(-effPriorLogOdds))
    scorestr = []
    scoresev = []
    P = PCA(dataset_train, labels_train, 5, eval=True)

    scorestr.append(logisticRegression(dataset_train, labels_train, dataset_train, labels_train, 0.1, ("Weighted quadratic", 10, 0.1), score=True, toCalibrate=True))
    scoresev.append(scoreCalibration(logisticRegression(dataset_train, labels_train, dataset_train, labels_train, 0.1, ("Weighted quadratic", 10, 0.1), score=True), labels_train, logisticRegression(dataset_train, labels_train, dataset_test, labels_test, 0.1, ("Weighted quadratic", 10, 0.1), score=True), labels_test, 0.1))
    print(0)
    scorestr.append(SupportVectorMachineKernelRBF(PCA(dataset_train, labels_train, 5), labels_train, PCA(dataset_train, labels_train, 5), labels_train, 0.5, ("SVMRBF", 1, 10, 0.001, 0.5), scoresFlag=True, toCalibrate=True))
    scoresev.append(SupportVectorMachineKernelRBF(PCA(dataset_train, labels_train, 5), labels_train, numpy.dot(P.transpose(), dataset_test), labels_test, 0.5, ("SVMRBF", 1, 10, 0.001, 0.5), scoresFlag=True, toCalibrate=True))
    print(1)
    scorestr.append(GMMclassify(PCA(dataset_train, labels_train, 5), labels_train, PCA(dataset_train, labels_train, 5), labels_train, 0.5, (("Tied", "Tied"), (32, 2), 0.01, 0.1), scores=True, toCalibrate=True))
    scoresev.append(GMMclassify(PCA(dataset_train, labels_train, 5), labels_train, numpy.dot(P.transpose(), dataset_test), labels_test, 0.5, (("Tied", "Tied"), (32, 2), 0.01, 0.1), scores=True, toCalibrate=True))
    print(2)

    gotscorestr = numpy.vstack(scorestr)
    gotscoresev = numpy.vstack(scoresev)
    fusion_labels = labels_test
    fusion_scores = modelFusion(gotscorestr, labels_train, gotscoresev, labels_test, 0.5)

    colors = ["r", "b", "g"]

    for j in range(3):
        plot_dcf = []
        plot_mindcf = []
        for i in range(pi_sign.size):
            cm = optimal_bayes_decisions(gotscoresev[j], fusion_labels, (pi_sign[i], 1, 1))
            plot_dcf.append(compute_bayes_risk(cm, (pi_sign[i], 1, 1))/compute_dummy_bayes((pi_sign[i], 1, 1)))
            plot_mindcf.append(compute_minDCF(gotscoresev[j], fusion_labels, (pi_sign[i], 1, 1)))

        plot_dcf = numpy.array(plot_dcf)
        plot_mindcf = numpy.array(plot_mindcf)

        plt.plot(effPriorLogOdds, plot_dcf, linestyle="solid", color=colors[j])
        plt.plot(effPriorLogOdds, plot_mindcf, linestyle="dashed", color=colors[j])

    print(4)
    plot_dcf = []
    plot_mindcf = []

    for i in range(pi_sign.size):
        cm = optimal_bayes_decisions(fusion_scores, fusion_labels, (pi_sign[i], 1, 1))
        plot_dcf.append(compute_bayes_risk(cm, (pi_sign[i], 1, 1))/compute_dummy_bayes((pi_sign[i], 1, 1)))
        plot_mindcf.append(compute_minDCF(fusion_scores, fusion_labels, (pi_sign[i], 1, 1)))

    plot_dcf = numpy.array(plot_dcf)
    plot_mindcf = numpy.array(plot_mindcf)

    plt.plot(effPriorLogOdds, plot_dcf, linestyle="solid", color="y")
    plt.plot(effPriorLogOdds, plot_mindcf, linestyle="dashed", color="y")
    plt.xlim([-3, 3])
    plt.xlabel("prior log-odds")
    plt.ylabel("Cprim")
    plt.legend(["W. Quad Log-Reg (actDCF)", "W. Quad Log-Reg (minDCF)", "SVM RBF (actDCF)", "SVM RBF (minDCF)", "GMM (actDCF)", "GMM (minDCF)", "Fusion (actDCF)", "Fusion (minDCF)"])
    plt.savefig('FusionPlotTest(Final-Calibrated-CorrectedWithPCA).svg')
    plt.show()'''

    # ----- Considerations on Candidate #1 (Weighed Quadratic Log-Reg) -----
    '''workingPoints = [(0.5, 1, 1), (0.1, 1, 1)]
    kfoldPlotMinDCFlambda(dataset_train, labels_train, k, workingPoints, [(logisticRegression, "Weighted quadratic Logistic Regression")], [("Weighted quadratic", 0.001, 0.1)])
    lambda_r = numpy.logspace(-5, 3, num=9)
    plot_mindcf = []

    for w in range(len(workingPoints)):
        piT = workingPoints[w][0]
        for l in range(lambda_r.size):
            scores = (logisticRegression(dataset_train, labels_train, dataset_test, labels_test, piT, ("Weighted quadratic", lambda_r[l], 0.1), True))
            plot_mindcf.append(compute_minDCF(scores, labels_test, workingPoints[w]))
    
    plot_mindcf = [(plot_mindcf[i]+plot_mindcf[lambda_r.size+i])/2 for i in range(lambda_r.size)]
    plt.plot(lambda_r, plot_mindcf, label="λ", color='#1f77b4')
    plt.xscale('log')
    plt.grid(True)
    plt.xlim([10 ** -5, 10 ** 3])
    plt.xlabel('λ')
    plt.ylabel('minCprim')
    plt.legend(["W. Quad. Log-Reg (π=0.1) [dev set]", "W. Quad. Log-Reg (π=0.1) [eval set]"])
    plt.savefig("VersusDevEval-WQuadLogReg.svg")
    plt.show()'''

    # ----- Considerations on Candidate #2 (SVM RBF) -----
    '''workingPoints = [(0.5, 1, 1), (0.1, 1, 1)]
    kfoldPlotMinDCFC(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoints, [(SupportVectorMachineKernelRBF, "Support Vector Maachine - KernelRBF")], [("SVMRBF", 1, 10, 10e-3, 0.5)])
    kfoldPlotMinDCFC(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoints,
                     [(SupportVectorMachineKernelRBF, "Support Vector Maachine - KernelRBF")],
                     [("SVMRBF", 1, 10, 10e-4, 0.5)])
    kfoldPlotMinDCFC(PCA(dataset_train, labels_train, 5), labels_train, k, workingPoints,
                     [(SupportVectorMachineKernelRBF, "Support Vector Maachine - KernelRBF")],
                     [("SVMRBF", 1, 10, 10e-5, 0.5)])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    P = PCA(dataset_train, labels_train, 5, eval=True)
    C = numpy.logspace(-5, 2, num=8)
    Gamma = [0.001, 0.0001, 0.00001]
    plot_mindcf = []
    for gamma in range(len(Gamma)):
        plot_mindcf = []
        for w in range(len(workingPoints)):
            piT = workingPoints[w][0]
            for c in range(C.size):
                scores = (SupportVectorMachineKernelRBF(PCA(dataset_train, labels_train, 5), labels_train, numpy.dot(P.transpose(), dataset_test), labels_test, piT, ("SVMRBF", C[c], Gamma[gamma], 0.5), scoresFlag=True))
                plot_mindcf.append(compute_minDCF(scores, labels_test, workingPoints[w]))
        plot_mindcf = [(plot_mindcf[i] + plot_mindcf[C.size + i]) / 2 for i in range(C.size)]
        plt.plot(C, plot_mindcf, label="C", color=colors[gamma])

    plt.xscale('log')
    plt.grid(True)
    plt.xlim([10 ** -5, 10 ** 2])
    plt.legend(["RBF γ=10e-3 [dev set]", "RBF γ=10e-4 [dev set]", "RBF γ=10e-5 [dev set]", "RBF γ=10e-3 [eval set]", "RBF γ=10e-4 [eval set]", "RBF γ=10e-5 [eval set]"])
    plt.xlabel("C")
    plt.ylabel("minCprim")
    plt.savefig("VersusDevEval-SVMRBF.svg")
    plt.show()
    '''

    # ----- Considerations on Candidate #3 (GMM) -----
    # to do







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
