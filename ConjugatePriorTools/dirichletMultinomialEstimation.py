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

#Find the log probability that we see a certain set of data
# give our prior.mate d
def dirichLogProb(priorList, uMatrix, vVector):
  K = len(uMatrix)
  total = 0.0
  for k in range(0, K):
    for i in range(0, len(uMatrix[k])):
      total += uMatrix[k][i]*math.log(priorList[k] + i)

  sumPrior = sum(priorList)
  for i in range(0, len(vVector)):
    total -= vVector[i] * math.log(sumPrior + i)

  return total

#Gives the derivative with respect to the log of prior.  This will be used to adjust the loss
def priorGradient(priorList, uMatrix, vVector):
	K = len(uMatrix)
	
	termToSubtract = 0
	for i in range(0, len(vVector)):
		termToSubtract += float(vVector[i]) / (sum(priorList) + i)
	
	retVal = [0]*K
	for j in range(0, K):
		for i in range(0, len(uMatrix[j])):
			retVal[j] += float(uMatrix[j][i]) / (priorList[j] + i)
	
	for j in range(0, K):
		retVal[j] -= termToSubtract
		
	return retVal

#The hessian is actually the sum of two matrices: a diagonal matrix and a constant-value matrix.
#We'll write two functions to get both
def priorHessianConst(priorList, vVector):
	total = 0
	for i in range(0, len(vVector)):
		total += float(vVector[i]) / (sum(priorList) + i)**2
	return total

def priorHessianDiag(priorList, uMatrix):
  K = len(uMatrix)
  retVal = [0]*K
  for k in range(0, K):
    for i in range(0, len(uMatrix[k])):
      retVal[k] -= uMatrix[k][i] / (priorList[k] + i)**2
  return retVal

	
# Compute the next value to try here
# http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf (eq 18)
def getPredictedStep(hConst, hDiag, gradient):
  K = len(gradient)
  numSum = 0.0
  for i in range(0, K):
    numSum += gradient[i] / hDiag[i]

  denSum = 0.0
  for i in range(0, K): denSum += 1.0 / hDiag[i]

  b = numSum / ((1.0/hConst) + denSum)

  retVal = [0]*K
  for i in range(0, K): retVal[i] = (b - gradient[i]) / hDiag[i]
  return retVal

# Uses the diagonal hessian on the log-alpha values	
def getPredictedStepAlt(hConst, hDiag, gradient, alphas):
  K = len(gradient)
  retVal = [0]*K

  denominators = [(gradient[k] - alphas[k]*hDiag[k]) for k in range(0, K)]
  for k in range(0, K):
    if (denominators[k] == 0): return retVal

  Z = 0
  for k in range(0, K):
    Z += alphas[k] / denominators[k]
  Z *= hConst

  Ss = [0]*K
  for k in range(0, K):
    Ss[k] = 1.0 / denominators[k] / (1 + Z)
  S = sum(Ss)

  for i in range(0, K): 
    retVal[i] = gradient[i] / denominators[k] * (1 - hConst * alphas[i] * S)

  return retVal

#The priors and data are global, so we don't need to pass them in
def getTotalLoss(trialPriors, uMatrix, vVector):
  return -1*dirichLogProb(trialPriors, uMatrix, vVector)
	
def predictStepUsingHessian(gradient, priors, uMatrix, vVector):
	totalHConst = priorHessianConst(priors, vVector)
	totalHDiag = priorHessianDiag(priors, uMatrix)		
	return getPredictedStep(totalHConst, totalHDiag, gradient)
	
def predictStepLogSpace(gradient, priors, uMatrix, vVector):
	totalHConst = priorHessianConst(priors, vVector)
	totalHDiag = priorHessianDiag(priors, uMatrix)
	return getPredictedStepAlt(totalHConst, totalHDiag, gradient, priors)
	

# Returns whether it's a good step, and the loss	
def testTrialPriors(trialPriors, uMatrix, vVector):
	for alpha in trialPriors: 
		if alpha <= 0: 
			return float("inf")
		
	return getTotalLoss(trialPriors, uMatrix, vVector)
	
def sqVectorSize(v):
	s = 0
	for i in range(0, len(v)): s += v[i] ** 2
	return s

def findDirichletPriors(uMatrix, vVector, initAlphas):
  priors = initAlphas

  # Let the learning begin!!
  #Only step in a positive direction, get the current best loss.
  currentLoss = getTotalLoss(priors, uMatrix, vVector)

  gradientToleranceSq = 2 ** -10
  learnRateTolerance = 2 ** -20

  count = 0
  while(count < 50):
    count += 1
    
    #Get the data for taking steps
    gradient = priorGradient(priors, uMatrix, vVector)
    gradientSize = sqVectorSize(gradient) 
    logging.debug("Iteration: %s Loss: %s ,Priors: %s, Gradient Size: %s" % (count, currentLoss, priors, gradientSize))
    
    if (gradientSize < gradientToleranceSq):
      logging.debug("Converged with small gradient")
      return priors
    
    trialStep = predictStepUsingHessian(gradient, priors, uMatrix, vVector)
    
    #First, try the second order method
    trialPriors = [0]*len(priors)
    for i in range(0, len(priors)): trialPriors[i] = priors[i] + trialStep[i]
    
    #TODO: Check for taking such a small step that the loss change doesn't register (essentially converged)
    #  Fix by ending
    
    loss = testTrialPriors(trialPriors, uMatrix, vVector)
    if loss < currentLoss:
      currentLoss = loss
      priors = trialPriors
      continue
    
    trialStep = predictStepLogSpace(gradient, priors, uMatrix, vVector)
    trialPriors = [0]*len(priors)
    for i in range(0, len(priors)): 
      try:
        trialPriors[i] = priors[i] * math.exp(trialStep[i])
      except:
        trialPriors[i] = priors[i]
    loss = testTrialPriors(trialPriors, uMatrix, vVector)

    #Step in the direction of the gradient until there is a loss improvement
    learnRate = 1.0
    while loss > currentLoss:
      learnRate *= 0.9
      trialPriors = [0]*len(priors)
      for i in range(0, len(priors)): trialPriors[i] = priors[i] + gradient[i]*learnRate
      loss = testTrialPriors(trialPriors, uMatrix, vVector)

    if (learnRate < learnRateTolerance):
      logging.debug("Converged with small learn rate")
      return priors

    currentLoss = loss
    priors = trialPriors
    
  logging.debug("Reached max iterations")
  return priors