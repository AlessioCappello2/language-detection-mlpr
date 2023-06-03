from utils import *
from DimensionalityReduction import *
from ModelValidation import *

def main():
    dataset_train, labels_train = load("./LanguageDetection/Train.txt")
    # dataset_test, labels_test = load("./LanguageDetection/Test.txt")
    k = 5
    workingPoint = (0.5, 1, 1)
    # pay attention to MVG_log and MVG
    classifiers = [(MVG_log, "Log-Multivariate Gaussian Classifier", "Default"), (MVG, "Naive Bayes", "Naive"),
                   (MVG, "Tied Covariance", "Tied"), (logisticRegression, "Logistic Regression", "Default"),
                   (logisticRegression, "Weighted Logistic Regression", "Weighted"), (logisticRegression, "Quadratic Logistic Regression", "Quadratic"),
                   (logisticRegression, "Weighted Quadratic Logistic Regression", "Weighted quadratic")]

    # show_histo(dataset_train, labels_train)
    # show_histo(zNormalization(dataset_train), labels_train)

    # PCA(dataset_train, labels_train)
    # PCA(zNormalization(dataset_train), labels_train)
    # PCA(centerDataset(dataset_train), labels_train)
    print("---------ORIGINAL DATA-------------\n")
    kfold(dataset_train, labels_train, k, workingPoint, classifiers)
    print("---------CENTERED DATA-------------\n")
    kfold(centerDataset(dataset_train), labels_train, k, workingPoint, classifiers)
    print("-------ZNORMALIZED DATA------------\n")
    kfold(zNormalization(dataset_train), labels_train, k, workingPoint, classifiers)
    print("--------NORMALIZED DATA------------\n")
    kfold(normalization(dataset_train), labels_train, k, workingPoint, classifiers)


if __name__ == '__main__':
    main()
