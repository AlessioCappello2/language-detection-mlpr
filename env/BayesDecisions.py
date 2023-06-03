import numpy

def compute_confusion_matrix(l, pl):
    confusion_matrix = numpy.zeros([2, 2])
    confusion_matrix[0, 0] = numpy.count_nonzero(numpy.where((l == 0) & (pl == 0), 1, 0))
    confusion_matrix[0, 1] = numpy.count_nonzero(numpy.where((l == 1) & (pl == 0), 1, 0))
    confusion_matrix[1, 0] = numpy.count_nonzero(numpy.where((l == 0) & (pl == 1), 1, 0))
    confusion_matrix[1, 1] = numpy.count_nonzero(numpy.where((l == 1) & (pl == 1), 1, 0))
    return confusion_matrix.astype(int)
      

def optimal_bayes_decisions(llr, labels, wp, brute_force=False, t=0):
    pi_t = wp[0]
    cfn = wp[1]
    cfp = wp[2]
    if not brute_force:
        t = -1 * numpy.log(pi_t*cfn/(1-pi_t)/cfp)

    predicted_labels = numpy.where(llr >= t, 1, 0)
    return compute_confusion_matrix(labels, predicted_labels)


def compute_bayes_risk(conf_mat, wp):
    pi_t = wp[0]
    cfn = wp[1]
    cfp = wp[2]
    FNR = conf_mat[0,1]/(conf_mat[0,1]+conf_mat[1,1])
    FPR = conf_mat[1,0]/(conf_mat[1,0]+conf_mat[0,0])
    return pi_t*cfn*FNR+(1-pi_t)*cfp*FPR


def compute_dummy_bayes(wp):
    pi_t = wp[0]
    cfn = wp[1]
    cfp = wp[2]
    return numpy.min(numpy.array([pi_t*cfn, (1-pi_t)*cfp]))


def compute_minDCF(llr, labels, wp):
    pi_t = wp[0]
    cfn = wp[1]
    cfp = wp[2]
    dcfmin = numpy.finfo(float).max
    den = compute_dummy_bayes(wp)
    FNR = []
    FPR = []

    for i in range(labels.size):
        cm = optimal_bayes_decisions(llr, labels, wp, True, llr[i])
        FNR.append(cm[0,1]/(cm[0,1]+cm[1,1]))
        FPR.append(cm[1,0]/(cm[1,0]+cm[0,0]))
        DCFu = compute_bayes_risk(cm, wp)
        DCF = DCFu/den
        if DCF < dcfmin:
            dcfmin = DCF

    # FNR = numpy.array(FNR)
    # FPR = numpy.array(FPR)

    # sorted_indices = numpy.argsort(FPR)
    # ROC_plot(FPR[sorted_indices], FNR[sorted_indices])

    return dcfmin