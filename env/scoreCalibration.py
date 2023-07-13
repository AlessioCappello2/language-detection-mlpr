import numpy
from LogisticRegressionModels import logisticRegression
from utils import vrow

def scoreCalibration(calibrationSet, calibrationLabels, uncalibratedScores, uncalibratedLabels, prior):
    scores = logisticRegression(vrow(calibrationSet), calibrationLabels, vrow(uncalibratedScores), uncalibratedLabels, prior, ("Weighted", 0.0001, prior), score=True)
    return scores - numpy.log(prior/(1-prior))

def modelFusion(scoresSet, scoresLabels, DTE, LTE, prior):
    scores = logisticRegression(scoresSet, scoresLabels, DTE, LTE, prior, ("Weighted", 0.0001, prior), score=True)
    return scores - numpy.log(prior/(1-prior))