import numpy 
import matplotlib.pyplot as plt
import scipy as sci
from utils import *

def PCA(dataset, labels, nd=3, graph=False, expl=False, eval=False):
    mu = vcol(dataset.mean(1))
    centered_dataset = numpy.round(dataset - mu, 2)
    covariance_matrix = numpy.dot(centered_dataset, centered_dataset.transpose()) / numpy.array(
        [centered_dataset.shape[1]])
    sigma, U = numpy.linalg.eigh(covariance_matrix)
    P = numpy.array(U[:, -1:-(nd+1):-1])
    if eval: return P
    y = numpy.dot(P.transpose(), dataset)  # dataset e non centered_ visto che vogliamo proiettare i dati originali
    if nd >= 2 and graph:
        plt.scatter(x=y[0, labels == 0], y=y[1, labels == 0], c="#5873E8",
                label='Non target')
        plt.scatter(x=y[0, labels == 1], y=y[1, labels == 1], c="#E35C3D",
                label='Target')
        plt.legend()
        # plt.savefig('scatter2d.svg')
        plt.show()
        show_histo(y, labels)

    if nd >= 3 and graph:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter3D(y[0, labels == 0], y[1, labels == 0], y[2, labels == 0], c="#5873E8", label='Non target')
        ax.scatter3D(y[0, labels == 1], y[1, labels == 1], y[2, labels == 1], c="#E35C3D", label='Target')
        plt.legend()
        plt.show()

    if expl:
        sigma = sigma[::-1]
        array = numpy.zeros(0)
        eigensum = numpy.sum(sigma)
        for i in range(7):
            sum = numpy.sum(sigma[:i])
            array = numpy.append(array, numpy.round(sum/eigensum, 2))

        plt.plot(numpy.arange(7), array)
        plt.xlabel('PCA components')
        plt.ylabel('Fraction of explained variance')
        plt.title('PCA - explained variance')
        plt.grid(True)
        plt.savefig('PCA_variance.svg')
        plt.show()

    return y


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
        plt.hist(lda_final[i, labels == 0], label='Non target', color="#5873E8", density=True, edgecolor='black',
                 linewidth=0.5, alpha=0.7)
        plt.hist(lda_final[i, labels == 1], label='Target', color="#E35C3D", density=True, edgecolor='black',
                 linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.tight_layout()
        # plt.savefig('lda.svg')
    plt.show()

    
def LDA(dataset, labels):
    mu = vcol(dataset.mean(1))
    m = 1
    SW, SB = get_within_between(dataset, labels, mu)
    LDA_scipy(dataset, labels, SB, SW, m)