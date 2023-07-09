import numpy
import matplotlib.pyplot as plt
import scipy as sci
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.metrics import confusion_matrix

def vcol(a):
    return a.reshape(a.size, 1)


def vrow(a):
    return a.reshape(1, a.size)


def load(filePath):
    labels = []
    dataset = []
    features = 0

    with open(filePath, "r") as language:
        for row in language:
            s = row.split(",")
            label = s.pop().split("\n").pop(0)
            dataset.append([float(i) for i in s[::]])
            labels.append(label)
            if features == 0:
                features = len(s)

    labels = numpy.array(labels, dtype=numpy.int32)
    dataset = numpy.array(dataset)
    dataset = dataset.reshape(int(dataset.size/features), features).transpose()

    return dataset, labels

def show_histo(d, l):
    d1 = d[:, l==1]
    d0 = d[:, l==0]

    for i in range(2):
        plt.figure()
        plt.hist(d0[i, :], label='Non target', color="#5873E8", density=True, edgecolor='black',
                 linewidth=0.5, alpha=0.7)
        plt.hist(d1[i, :], label='Target', color="#E35C3D", density=True, edgecolor='black',
                 linewidth=0.5, alpha=0.7)
        plt.legend()

        plt.tight_layout()
        # plt.savefig('component_%d.svg' % i)
        plt.show()


def centerDataset(d):
    return d - vcol(d.mean(1))


def zNormalization(d):
    return (d - vcol(d.mean(1)))/(vcol(numpy.std(d, axis=1)))


def normalization(d):
    min_vals = vcol(numpy.min(d, axis=1))
    max_vals = vcol(numpy.max(d, axis=1))
    den = max_vals - min_vals
    return (d-min_vals)/den


def heatmap(dataset, cmap):
    cor_matrix = numpy.corrcoef(dataset)
    sns.heatmap(cor_matrix, cmap=cmap)
    plt.savefig('heatmap_%s.svg' % cmap)
    plt.show()

