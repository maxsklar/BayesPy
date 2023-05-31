#!/usr/bin/python
#
# A library for finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

import random
import logging
import numpy as np
from scipy.special import digamma, polygamma

def trigamma(x): return polygamma(1, x)


def dirichletLogProb(priorList, data, hyperPrior):
    K = data.K
    total = 0.0

    for k in range(K):
        total += np.sum(data.U[k] * np.log(priorList[k] + np.arange(len(data.U[k]))))

    sumPrior = np.sum(priorList)
    total -= np.sum(data.V * np.log(sumPrior + np.arange(len(data.V))))

    total += np.sum(priorList) * hyperPrior
    return total


def priorGradient(priorList, data, hyperPrior=0):
    K = data.K

    termToSubstract = np.sum(data.V / (np.sum(priorList) + np.arange(len(data.V))))

    retVal = np.zeros(K)
    for j in range(K):
        retVal[j] = np.sum(data.U[j] / (priorList[j] + np.arange(len(data.U[j]))))

    retVal -= termToSubstract
    retVal += hyperPrior * np.sum(priorList)

    return retVal


def priorHessianConst(priorList, data):
    total = np.sum(data.V / (np.sum(priorList) + np.arange(len(data.V))) ** 2)
    return total


def priorHessianDiag(priorList, data):
    K = len(data.U)
    retVal = np.zeros(K)
    for k in range(K):
        retVal[k] = -np.sum(data.U[k] / (priorList[k] + np.arange(len(data.U[k]))) ** 2)
    return retVal


# Compute the next value to try here
# http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf (eq 18)
# https://www.researchgate.net/profile/Max-Sklar/publication/370927162_Algorithms_for_Multivariate_Newton-Raphson_for_Optimization/links/64698c0170202663165fd82a/Algorithms-for-Multivariate-Newton-Raphson-for-Optimization.pdf
def getPredictedStep(hConst, hDiag, gradient):
    K = len(gradient)
    numSum = np.sum(gradient / hDiag)
    denSum = np.sum(1.0 / hDiag)
    b = numSum / ((1.0 / hConst) + denSum)
    return [(b - gradient[k]) / hDiag[k] for k in range(K)]


# Uses the diagonal hessian on the log-alpha values
# https://www.researchgate.net/profile/Max-Sklar/publication/370927162_Algorithms_for_Multivariate_Newton-Raphson_for_Optimization/links/64698c0170202663165fd82a/Algorithms-for-Multivariate-Newton-Raphson-for-Optimization.pdf
def getPredictedStepAlt(hConst, hDiag, gradient, alphas):
    x = gradient + alphas * hDiag
    Z = 1.0 / hConst + np.sum(alphas / x)
    S = np.sum(alphas * gradient / x)
    return (S / Z - gradient) / x


# The priors and data are global, so we don't need to pass them in
def getTotalLoss(trialPriors, data, hyperPrior=0):
    return -1 * dirichletLogProb(trialPriors, data, hyperPrior)


def predictStepUsingHessian(gradient, priors, data):
    totalHConst = priorHessianConst(priors, data)
    totalHDiag = priorHessianDiag(priors, data)
    return getPredictedStep(totalHConst, totalHDiag, gradient)


def predictStepLogSpace(gradient, priors, data):
    totalHConst = priorHessianConst(priors, data)
    totalHDiag = priorHessianDiag(priors, data)
    return getPredictedStepAlt(totalHConst, totalHDiag, gradient, priors)


def sqVectorSize(v):
    return np.sum(v ** 2)


class CompressedRowData:
    def __init__(self, K):
        self.K = K
        self.V = []
        self.U = [np.array([]) for _ in range(K)]

    def appendRow(self, row, weight):
        if len(row) != self.K:
            logging.error("row must have K=" + str(self.K) + " counts")

        for k in range(self.K):
            elemK = row[k]
            if len(self.U[k]) < elemK:
                self.U[k] = np.append(self.U[k], np.zeros(elemK - len(self.U[k])))
            self.U[k][:elemK] += weight

        rowSum = sum(row)
        if len(self.V) < rowSum:
            self.V = np.append(self.V, np.zeros(rowSum - len(self.V)))
        self.V[:rowSum] += weight


def testTrialPriors(trialPriors, data, hyperPrior):
    if np.any(trialPriors < 0):
        return float("inf")
    return getTotalLoss(trialPriors, data, hyperPrior)


def findDirichletPriors(data, initAlphas, iterations, hyperPrior=0):
    priors = np.asarray(initAlphas)

    # Let the learning begin!!
    # Only step in a positive direction, get the current best loss.
    currentLoss = getTotalLoss(priors, data, hyperPrior)

    gradientToleranceSq = 2 ** -10
    learnRateTolerance = 2 ** -20

    count = 0
    while count < iterations:
        count += 1

        # Get the data for taking steps
        gradient = priorGradient(priors, data, hyperPrior)
        gradientSize = sqVectorSize(gradient)
        logging.debug("Iteration: %s Loss: %s ,Priors: %s, Gradient Size: %s" % (count,
                                                                                 currentLoss,
                                                                                 priors,
                                                                                 gradientSize))

        if gradientSize < gradientToleranceSq:
            logging.debug("Converged with small gradient")
            return priors

        trialStep = predictStepUsingHessian(gradient, priors, data)

        # First, try the second order method
        trialPriors = np.zeros_like(priors)
        trialPriors = priors + trialStep

        # @@TODO: Check for taking such a small step that the loss change doesn't register (essentially converged)
        #  Fix by ending

        loss = testTrialPriors(trialPriors, data, hyperPrior)
        if loss < currentLoss:
            currentLoss = loss
            priors = trialPriors
            continue

        trialStep = predictStepLogSpace(gradient, priors, data)
        trialPriors = priors * np.exp(trialStep)

        loss = testTrialPriors(trialPriors, data, hyperPrior)

        # Step in the direction of the gradient until there is a loss improvement
        learnRate = 1.0
        while loss > currentLoss:
            learnRate *= 0.9
            trialPriors = priors + gradient * learnRate
            loss = testTrialPriors(trialPriors, data, hyperPrior)

        if learnRate < learnRateTolerance:
            logging.debug("Converged with small learn rate")
            return priors

        currentLoss = loss
        priors = trialPriors

    logging.debug("Reached max iterations")
    return priors
