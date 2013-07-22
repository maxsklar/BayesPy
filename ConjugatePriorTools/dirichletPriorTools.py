#!/usr/bin/python
#
# A library for finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2013 Max Sklar

import math
import random

#Find the log probability that we see a certain set of data
# give our prior.
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
  for i in range(0, K): 
    term = alphas[i] * (hDiag[i] + hConst)
    retVal[i] = -1* gradient[i] / (gradient[i] + term)
  return retVal

#The priors and data are global, so we don't need to pass them in
def getTotalLoss(trialPriors, uMatrix, vVector):
  return -1*dirichLogProb(trialPriors, uMatrix, vVector)
	
def predictStepUsingHessian(gradient, priors, uMatrix, vVector):
	totalHConst = priorHessianConst(priors, vVector)
	totalHDiag = priorHessianDiag(priors, uMatrix)		
	return getPredictedStep(totalHConst, totalHDiag, gradient)
	
def predictStepUsingDiagHessian(gradient, priors, uMatrix, vVector):
	totalHConst = priorHessianConst(priors, vVector)
	totalHDiag = priorHessianDiag(priors, uMatrix)
	return getPredictedStepAlt(totalHConst, totalHDiag, gradient, priors)
	

# Returns whether it's a good step, and the loss	
def testTrialPriors(trialPriors, uMatrix, vVector):
	n = len(trialPriors)
	for i in range(0, n): 
		if trialPriors[i] <= 0: 
			return float("inf")
		
	return getTotalLoss(trialPriors, uMatrix, vVector)
	
def sqVectorSize(v):
	s = 0
	for i in range(0, len(v)): s += v[i] ** 2
	return s

def findDirichletPriors(uMatrix, vVector, initAlphas, verbose):
  priors = initAlphas

  # Let the learning begin!!
  #Only step in a positive direction, get the current best loss.
  currentLoss = getTotalLoss(priors, uMatrix, vVector)

  gradientToleranceSq = 2 ** -20

  count = 0
  while(count < 10000):
    count += 1
    if (verbose): print  count, "Loss: ", currentLoss, ", Priors: ", priors
    
    #Get the data for taking steps
    gradient = priorGradient(priors, uMatrix, vVector)
    if (sqVectorSize(gradient) < gradientToleranceSq):
      if (verbose): print "Converged with small gradient"
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
    
    trialStep = predictStepUsingDiagHessian(gradient, priors, uMatrix, vVector)
    trialPriors = [0]*len(priors)
    for i in range(0, len(priors)): trialPriors[i] = priors[i] * math.exp(trialStep[i])
    loss = testTrialPriors(trialPriors, uMatrix, vVector)
    if loss < currentLoss:
      currentLoss = loss
      priors = trialPriors
      continue
    
    if (verbose): print "Converged with no loss improvement"
    return priors
  if (verbose): print "Reached max iterations"
  return priors


def getLossForMultinomials(multinomials):

def findDirichletPriorsFromMultinomials(multinomials, initAlphas, verbose):  
  priors = initAlphas
  
  K = len(priors)
  N = len(multinomials)
  
  # Compute the sufficient statistic
  ss = [0]*K
  for n in range(0, N):
    for k in range(0, K):
      ss[k] += math.log(mutlinomials[n][k])
  
  