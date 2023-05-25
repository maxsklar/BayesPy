#!/usr/bin/python
#
# A library for finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2013 Max Sklar

import math
import logging
import random
import scipy.special as mathExtra

def digamma(x): return float(mathExtra.psi(x))
def trigamma(x): return float(mathExtra.polygamma(1, x))
  

# Find the "sufficient statistic" for a group of multinomials.
# Essential, it's the average of the log probabilities
def getSufficientStatistic(multinomials):
  N = len(multinomials)
  K = len(multinomials[0])

  retVal = [0]*K

  for m in multinomials:
    for k in range(0, K):
      retVal[k] += math.log(m[k])

  for k in range(0, K): retVal[k] /= N
  return retVal

# Find the log probability of the data for a given dirichlet
# This is equal to the log probabiliy of the data.. up to a linear transform
def logProbForMultinomials(alphas, ss):
  retVal = math.lgamma(sum(alphas))
  retVal -= sum(map(math.lgamma, alphas))
  retVal += sum(p*q for p,q in zip(alphas, ss))
  return retVal

#Gives the derivative with respect to the log of prior.  This will be used to adjust the loss
def getGradientForMultinomials(alphas, ss):
  K = len(alphas)
  C = digamma(sum(alphas))
  retVal = [C]*K
  for k in range(0, K):
    retVal[k] += ss[k] - digamma(alphas[k])
  return retVal

#The hessian is actually the sum of two matrices: a diagonal matrix and a constant-value matrix.
#We'll write two functions to get both
def priorHessianConst(alphas, ss): return -trigamma(sum(alphas))
def priorHessianDiag(alphas, ss): return [trigamma(a) for a in alphas]
	
# Compute the next value to try here
# http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf (eq 18)
# https://www.researchgate.net/profile/Max-Sklar/publication/370927162_Algorithms_for_Multivariate_Newton-Raphson_for_Optimization/links/64698c0170202663165fd82a/Algorithms-for-Multivariate-Newton-Raphson-for-Optimization.pdf
def getPredictedStep(hConst, hDiag, gradient):
  K = len(gradient)
  numSum = sum(gradient[k] / hDiag[k] for k in range(K))
  denSum = sum(1.0 / hDiag[k] for k in range(K))
  b = numSum / ((1.0/hConst) + denSum)
  return [(b - gradient[k]) / hDiag[k] for k in range(K)]

# Uses the diagonal hessian on the log-alpha values
# https://www.researchgate.net/profile/Max-Sklar/publication/370927162_Algorithms_for_Multivariate_Newton-Raphson_for_Optimization/links/64698c0170202663165fd82a/Algorithms-for-Multivariate-Newton-Raphson-for-Optimization.pdf
def getPredictedStepAlt(hConst, hDiag, gradient, alphas):
  x = [(grad + alpha * h) for grad, alpha, h in zip(gradient, alphas, hDiag)]
  Z = 1.0 / hConst + sum(alpha / xi for alpha, xi in zip(alphas, x))
  S = sum(alpha * grad / xi for alpha, grad, xi in zip(alphas, gradient, x))
  return [(S / Z - grad) / xi for grad, xi in zip(gradient, x)]

#The priors and data are global, so we don't need to pass them in
def getTotalLoss(trialPriors, ss):
  return -1*logProbForMultinomials(trialPriors, ss)
	
def predictStepUsingHessian(gradient, priors, ss):
	totalHConst = priorHessianConst(priors, ss)
	totalHDiag = priorHessianDiag(priors, ss)		
	return getPredictedStep(totalHConst, totalHDiag, gradient)
	
def predictStepLogSpace(gradient, priors, ss):
	totalHConst = priorHessianConst(priors, ss)
	totalHDiag = priorHessianDiag(priors, ss)
	return getPredictedStepAlt(totalHConst, totalHDiag, gradient, priors)
	

# Returns whether it's a good step, and the loss	
def testTrialPriors(trialPriors, ss):
	for alpha in trialPriors: 
		if alpha <= 0: 
			return float("inf")
		
	return getTotalLoss(trialPriors, ss)
	
def sqVectorSize(v):
	s = 0
	for i in range(0, len(v)): s += v[i] ** 2
	return s

def findDirichletPriors(ss, initAlphas):
  priors = initAlphas

  # Let the learning begin!!
  #Only step in a positive direction, get the current best loss.
  currentLoss = getTotalLoss(priors, ss)

  gradientToleranceSq = 2 ** -20
  learnRateTolerance = 2 ** -10

  count = 0
  while(count < 1000):
    count += 1
    
    #Get the data for taking steps
    gradient = getGradientForMultinomials(priors, ss)
    gradientSize = sqVectorSize(gradient) 
    logging.debug(count, "Loss: ", currentLoss, ", Priors: ", priors, ", Gradient Size: ", gradientSize, gradient)
    
    if (gradientSize < gradientToleranceSq):
      logging.debug("Converged with small gradient")
      return priors
    
    trialStep = predictStepUsingHessian(gradient, priors, ss)
    
    #First, try the second order method
    trialPriors = [0]*len(priors)
    for i in range(0, len(priors)): trialPriors[i] = priors[i] + trialStep[i]
    
    loss = testTrialPriors(trialPriors, ss)
    if loss < currentLoss:
      currentLoss = loss
      priors = trialPriors
      continue
    
    trialStep = predictStepLogSpace(gradient, priors, ss)
    trialPriors = [0]*len(priors)
    for i in range(0, len(priors)): trialPriors[i] = priors[i] * math.exp(trialStep[i])
    loss = testTrialPriors(trialPriors, ss)

    #Step in the direction of the gradient until there is a loss improvement
    loss = 10000000
    learnRate = 1.0
    while loss > currentLoss:
      learnRate *= 0.9
      trialPriors = [0]*len(priors)
      for i in range(0, len(priors)): trialPriors[i] = priors[i] + gradient[i]*learnRate
      loss = testTrialPriors(trialPriors, ss)

    if (learnRate < learnRateTolerance):
      logging.debug("Converged with small learn rate")
      return priors

    currentLoss = loss
    priors = trialPriors
    
  logging.debug("Reached max iterations")
  return priors

def findDirichletPriorsFromMultinomials(multinomials, initAlphas):
	ss = getSufficientStatistic(multinomials)
	return findDirichletPriors(ss, initAlphas)