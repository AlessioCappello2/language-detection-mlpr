from utils import *
from DimensionalityReduction import *
from ModelValidation import *

def main():
    dataset_train, labels_train = load("./LanguageDetection/Train.txt")
    # dataset_test, labels_test = load("./LanguageDetection/Test.txt")
    k = 5
    workingPoint = (0.5, 1, 1)
    # PCA(dataset_train, labels_train)
    # LDA(dataset_train, labels_train)
    kfold(dataset_train, labels_train, k, workingPoint)


if __name__ == '__main__':
    main()
