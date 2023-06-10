from utils import *
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

    parametersSVM = [("SVML", 1.0, 1.0), ("SVMP", 1.0, 1.0, 1.0, 2), ("SVMRBF", 1.0, 1.0, 10.0)]

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

    # show_histo(dataset_train, labels_train)
    # show_histo(zNormalization(dataset_train), labels_train)

    # PCA(dataset_train, labels_train)
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

    kfold(dataset_train, labels_train, k, workingPoint, classifiersGMM, parametersGMM)

    # kfoldPlotMinDCFlambda(dataset_train, labels_train, k, workingPoint, classifiersLR, parametersLR)
    # kfoldBayesErrorPlot(dataset_train, labels_train, k, workingPoint, logisticRegression, ("Quadratic", 0.001))
    # kfoldBayesErrorPlot(dataset_train, labels_train, k, workingPoint, SupportVectorMachineKernelRBF, ("SVMRBF", 1.0, 1.0, 10.0))

if __name__ == '__main__':
    main()
