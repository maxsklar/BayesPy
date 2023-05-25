#!/usr/bin/python
#
# A library for finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2013 Max Sklar

import math
import random
import logging
import scipy.special as mathExtra

def digamma(x): return float(mathExtra.psi(x))
def trigamma(x): return float(mathExtra.polygamma(1, x))

#Find the log probability that we see a certain set of data
# give our prior.mate d
def dirichLogProb(priorList, data, Beta = None, W = None):
  K = data.K
  total = 0.0
  for k in range(0, K):
    for i in range(0, len(data.U[k])):
      total += data.U[k][i]*math.log(priorList[k] + i)

  sumPrior = sum(priorList)
  for i in range(0, len(data.V)):
    total -= data.V[i] * math.log(sumPrior + i)

  # Add prior
  if (Beta != None):
    for i in range(0, K):
      total -= priorList[i]*Beta[i]

  if (W != None):
    total += W*math.lgamma(sumPrior)
    for k in range(0, K): total -= W*(math.lgamma(priorList[k]))
  return total

#Gives the derivative with respect to the prior.  This will be used to adjust the loss
def priorGradient(priorList, data, Beta = None, W = None):
	K = data.K
	
	termToSubtract = 0
	for i in range(0, len(data.V)):
		termToSubtract += float(data.V[i]) / (sum(priorList) + i)
	
	retVal = [0]*K
	for j in range(0, K):
		for i in range(0, len(data.U[j])):
			retVal[j] += float(data.U[j][i]) / (priorList[j] + i)
	
	for j in range(0, K):
		retVal[j] -= termToSubtract
	
	# Add Prior
	if (Beta != None):
		for k in range(0, K):
			retVal[k] -= Beta[k]
	
	if (W != None):
		for k in range(0, K):
			retVal[k] += W*(digamma(sum(priorList)) - digamma(priorList[k]))
		
	return retVal

#The hessian is actually the sum of two matrices: a diagonal matrix and a constant-value matrix.
#We'll write two functions to get both
def priorHessianConst(priorList, data, W = None):
	total = 0
	for i in range(0, len(data.V)):
		total += float(data.V[i]) / (sum(priorList) + i)**2
	if (W != None):
		total += W*trigamma(sum(priorList))
	return total

def priorHessianDiag(priorList, data, W = None):
  K = len(data.U)
  retVal = [0]*K
  for k in range(0, K):
    for i in range(0, len(data.U[k])):
      retVal[k] -= data.U[k][i] / (priorList[k] + i)**2
  if (W != None):
    for k in range(0, K):
      retVal[k] -= W*trigamma(priorList[k])
  return retVal

	
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
def getTotalLoss(trialPriors, data, Beta = None, W = None):
  return -1*dirichLogProb(trialPriors, data, Beta = None, W = None)
	
def predictStepUsingHessian(gradient, priors, data, W = None):
	totalHConst = priorHessianConst(priors, data, W)
	totalHDiag = priorHessianDiag(priors, data, W)
	return getPredictedStep(totalHConst, totalHDiag, gradient)
	
def predictStepLogSpace(gradient, priors, data, W = None):
	totalHConst = priorHessianConst(priors, data, W)
	totalHDiag = priorHessianDiag(priors, data, W)
	return getPredictedStepAlt(totalHConst, totalHDiag, gradient, priors)
	

# Returns whether it's a good step, and the loss	
def testTrialPriors(trialPriors, data, Beta = None, W = None):
	for alpha in trialPriors: 
		if alpha <= 0: 
			return float("inf")
		
	return getTotalLoss(trialPriors, data, Beta, W)
	
def sqVectorSize(v):
	s = 0
	for i in range(0, len(v)): s += v[i] ** 2
	return s

class CompressedRowData:
  def __init__(self, K):
    self.K = K
    self.V = []
    self.U = []
    for k in range(0, K): self.U.append([])
    
  def appendRow(self, row, weight):
    if (len(row) != self.K): logging.error("row must have K=" + str(self.K) + " counts")
    
    for k in range(0, self.K): 
      for j in range(0, row[k]):
        if (len(self.U[k]) == j): self.U[k].append(0)
        self.U[k][j] += weight
      
    for j in range(0, sum(row)):
      if (len(self.V) == j): self.V.append(0)
      self.V[j] += weight
  

def findDirichletPriors(data, initAlphas, iterations, Beta = None, W = None):
  priors = initAlphas

  # Let the learning begin!!
  #Only step in a positive direction, get the current best loss.
  currentLoss = getTotalLoss(priors, data, Beta, W)

  gradientToleranceSq = 2 ** -10
  learnRateTolerance = 2 ** -20

  count = 0
  while(count < iterations):
    count += 1
    
    #Get the data for taking steps
    gradient = priorGradient(priors, data, Beta, W)
    gradientSize = sqVectorSize(gradient) 
    logging.debug("Iteration: %s Loss: %s ,Priors: %s, Gradient Size: %s" % (count, currentLoss, priors, gradientSize))
    
    if (gradientSize < gradientToleranceSq):
      logging.debug("Converged with small gradient")
      return priors
    
    trialStep = predictStepUsingHessian(gradient, priors, data, W)
    
    #First, try the second order method
    trialPriors = [0]*len(priors)
    for i in range(0, len(priors)): trialPriors[i] = priors[i] + trialStep[i]
    
    #TODO: Check for taking such a small step that the loss change doesn't register (essentially converged)
    #  Fix by ending
    
    loss = testTrialPriors(trialPriors, data, Beta, W)
    if loss < currentLoss:
      currentLoss = loss
      priors = trialPriors
      continue
    
    trialStep = predictStepLogSpace(gradient, priors, data, W)
    trialPriors = [0]*len(priors)
    for i in range(0, len(priors)): 
      try:
        trialPriors[i] = priors[i] * math.exp(trialStep[i])
      except:
        trialPriors[i] = priors[i]
    loss = testTrialPriors(trialPriors, data, Beta, W)

    #Step in the direction of the gradient until there is a loss improvement
    learnRate = 1.0
    while loss > currentLoss:
      learnRate *= 0.9
      trialPriors = [0]*len(priors)
      for i in range(0, len(priors)): trialPriors[i] = priors[i] + gradient[i]*learnRate
      loss = testTrialPriors(trialPriors, data, Beta, W)

    if (learnRate < learnRateTolerance):
      logging.debug("Converged with small learn rate")
      return priors

    currentLoss = loss
    priors = trialPriors
    
  logging.debug("Reached max iterations")
  return priors