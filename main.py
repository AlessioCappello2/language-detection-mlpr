from utils import *
from DimensionalityReduction import *
from ModelValidation import *

def main():
    dataset_train, labels_train = load("./LanguageDetection/Train.txt")
    # dataset_test, labels_test = load("./LanguageDetection/Test.txt")
    k = 5
    workingPoint = (0.5, 1, 1)
    K = 1.0
    C = 1.0
    c = 1.0
    d = 3
    gamma = 10.0
    # pay attention to MVG_log and MVG
    classifiers = [(MVG_log, "Log-Multivariate Gaussian Classifier"), (MVG, "Naive Bayes"),
                   (MVG, "Tied Covariance"), (logisticRegression, "Logistic Regression"),
                   (logisticRegression, "Weighted Logistic Regression"), (logisticRegression, "Quadratic Logistic Regression"),
                   (logisticRegression, "Weighted Quadratic Logistic Regression"),
                   (SupportVectorMachineLinear, "Support Vector Machine - Linear"),
                   (SupportVectorMachineKernelPoly, "Support Vector Machine - Kernel Poly"),
                   (SupportVectorMachineKernelRBF, "Support Vector Machine - Kernel RBF")]

    parameters = [("Default"), ("Naive"), ("Tied"), ("Default", 0.001), ("Weighted", 0.001), ("Quadratic", 0.001),
                  ("Weighted quadratic", 0.001), ("SVML", K, C), ("SVMP", K, C, c, d), ("SVMRBF", K, C, gamma)]

    # show_histo(dataset_train, labels_train)
    # show_histo(zNormalization(dataset_train), labels_train)

    # PCA(dataset_train, labels_train)
    # PCA(zNormalization(dataset_train), labels_train)
    # PCA(centerDataset(dataset_train), labels_train)
    print("---------ORIGINAL DATA-------------\n")
    kfold(dataset_train, labels_train, k, workingPoint, classifiers, parameters)
    # print("---------CENTERED DATA-------------\n")
    # kfold(centerDataset(dataset_train), labels_train, k, workingPoint, classifiers, parameters)
    # print("-------ZNORMALIZED DATA------------\n")
    # kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiers, parameters)
    # print("--------NORMALIZED DATA------------\n")
    # kfold(normalization(dataset_train), labels_train, k, workingPoint, classifiers, parameters)


if __name__ == '__main__':
    main()
