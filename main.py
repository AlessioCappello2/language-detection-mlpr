from utils import *
from DimensionalityReduction import *

def main():
    dataset_train, labels_train = load("./LanguageDetection/Train.txt")
    # dataset_test, labels_test = load("./LanguageDetection/Test.txt")
    # PCA(dataset_train, labels_train)
    LDA(dataset_train, labels_train)


if __name__ == '__main__':
    main()
