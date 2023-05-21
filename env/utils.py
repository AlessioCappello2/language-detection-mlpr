import numpy
import matplotlib.pyplot as plt
import scipy as sci
from mpl_toolkits import mplot3d
from sklearn.metrics import confusion_matrix

def vcol(a):
    return a.reshape(a.size, 1)


def vrow(a):
    return a.reshape(1, a.size)


def load(filePath):
    labels = []
    dataset = []
    elements_per_row = 0

    with open(filePath, "r") as language:
        for row in language:
            s = row.split(",")
            label = s.pop().split("\n").pop(0)
            dataset.append([float(i) for i in s[::]])
            labels.append(label)
            if elements_per_row == 0:
                elements_per_row = len(s)

    labels = numpy.array(labels, dtype=numpy.int32)
    dataset = numpy.array(dataset)
    dataset = dataset.reshape(int(dataset.size/elements_per_row), elements_per_row).transpose()

    return dataset, labels

def show_histo(d, l):
    d1 = d[:, l==1]
    d0 = d[:, l==0]

    print(d0)
    print(d1)

    for i in range(6):
        plt.figure()
        plt.hist(d0[i, :], label='Non italian')
        plt.hist(d1[i, :], label='Italian')
        plt.legend()

        plt.tight_layout()
        plt.savefig('hist_%d.pdf' % i)
    plt.show()