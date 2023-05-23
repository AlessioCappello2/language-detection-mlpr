import numpy 
import matplotlib.pyplot as plt
import scipy as sci
from utils import *

def PCA(dataset, labels):
    mu = vcol(dataset.mean(1))
    centered_dataset = numpy.round(dataset - mu, 2)
    covariance_matrix = numpy.dot(centered_dataset, centered_dataset.transpose()) / numpy.array(
        [centered_dataset.shape[1]])
    sigma, U = numpy.linalg.eigh(covariance_matrix)
    P = numpy.array(U[:, -1:-4:-1])
    y = numpy.dot(P.transpose(), dataset)  # dataset e non centered_ visto che vogliamo proiettare i dati originali

    plt.scatter(x=y[0, labels == 0], y=y[1, labels == 0], c="#1968E3",
               label='Non italian')
    plt.scatter(x=y[0, labels == 1], y=y[1, labels == 1], c="#E3BF19",
               label='Italian')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(y[0, labels == 0], y[1, labels == 0], y[2, labels == 0], label='Non italian')
    ax.scatter3D(y[0, labels == 1], y[1, labels == 1], y[2, labels==1], label='Italian')
    plt.legend()
    plt.show()


def get_within_between(dataset, labels, mu):
    mu = vcol(mu)
    class_means = []
    covariance_matrixes = []
    SW = numpy.zeros((6, 6))
    SB = numpy.zeros((6, 6))
    for i in range(2):
        class_means.append(dataset[:, labels == i].mean(1))
        centered_dataset = dataset[:, labels == i] - vcol(numpy.array(class_means[i]))
        covariance_matrixes.append(numpy.dot(centered_dataset, centered_dataset.transpose()) /
                                   centered_dataset.shape[1])
        SW += (covariance_matrixes[i] * centered_dataset.shape[1])
        SB += numpy.dot(vcol(numpy.array(class_means[i]))-mu, (vcol(numpy.array(class_means[i]))-mu).transpose() *
                        centered_dataset.shape[1])

    SW /= dataset.shape[1]
    SB /= dataset.shape[1]
    return SW, SB
    

def LDA_scipy(dataset, labels, SB, SW, m):
    s, B = sci.linalg.eigh(SB, SW)  # s autovalori, B autovettori
    lda_final = numpy.dot(B[:, -1:-2:-1].T, dataset)

    for i in range(1):
        plt.figure()
        plt.hist(lda_final[i, labels==0], label='Non italian', density=True)
        plt.hist(lda_final[i, labels==1], label='Italian', density=True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('hist_%d.pdf' % i)
    plt.show()
    
    
def LDA(dataset, labels):
    mu = vcol(dataset.mean(1))
    m = 1
    SW, SB = get_within_between(dataset, labels, mu)
    LDA_scipy(dataset, labels, SB, SW, m)