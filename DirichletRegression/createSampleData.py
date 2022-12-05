#!/usr/bin/python
#
# First Argument: number of multinomials to draw
# Second Argument: number of points to draw from each mutlinomial
# The rest of the arguments: the alpha parameters for the dirichlet 

import sys
import csv
import math
import random
import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option('-K', '--K', dest='K', default='2', help='Number of classes')
parser.add_option('-N', '--N', dest='N', help='Number of datapoints')
parser.add_option('-F', '--F', dest='F', help='Number of features to create')

(options, args) = parser.parse_args()
numDatapoints = int(options.N)
numFeatures = int(options.F)
numCategories = int(options.K)

def drawCategory(distribution):
  K = len(distribution)
  
  r = sum(distribution) * random.random()
  runningTotal = 0
  for k in range(0, K):
    runningTotal += distribution[k]
    if (r < runningTotal): return k
  
  return K-1

# Returns a discrete distribution
def drawFromDirichlet(log_alphas):
  K = len(log_alphas)

  highest_score = max(log_alphas)
  alphas_fixed = np.zeros(K)
  for k in range(K): alphas_fixed[k] = math.exp(log_alphas[k] - highest_score)
  mean_multinomial = np.zeros(numCategories)
  for k in range(K): mean_multinomial[k] = alphas_fixed[k] / sum(alphas_fixed)

  if (highest_score > 50): return mean_multinomial

  alphas = np.zeros(numCategories)
  for k in range(numCategories):
    alphas[k] = math.exp(log_alphas[k])

  if (sum(alphas) < 10**(-3)):
    category = drawCategory(mean_multinomial)
    retval = np.zeros(K)
    retval[category] = 1
    return retval

  multinomial = np.zeros(K)
  for i in range(0, K):
    if (alphas[i] > 0):
      multinomial[i] = random.gammavariate(alphas[i], 1)
  S = sum(multinomial)

  for k in range(K): multinomial[k] /= S
  return multinomial

def sampleFromMultinomial(multinomial, M):
  buckets = np.zeros(len(multinomial), dtype=np.int16)
  
  for m in range(0, M):
    category = drawCategory(multinomial)
    buckets[category] += 1
    
  return buckets

# Generate the "correct" feature weights randomly from a cauchy distribution

weights = np.zeros((numFeatures, numCategories))
featureFrequency = np.zeros(numFeatures)

feature_list_file_name = "SampleData/featureList.txt"
training_set_file_name = "SampleData/trainingSet.txt"
weights_file_name = "SampleData/weights.txt"

feature_list_file = open(feature_list_file_name, "w")
training_set_file = open(training_set_file_name, "w")
weights_file = open(weights_file_name, "w")

for f in range(numFeatures):
  for k in range(numCategories):
    weights[f][k] = np.random.normal(0, 1)
    weights_file.write(str(weights[f][k]))
    weights_file.write("\t")
  weights_file.write("\n")
  feature_list_file.write(str(f))
  feature_list_file.write("\n")

  # Use Ziph's Law (sort of)
  featureFrequency[f] = 1.0 / (f + 1.0)

#Generate DataPoints
for n in range(numDatapoints):
  features = []
  scores = np.zeros(numCategories)
  for f in range(numFeatures):
    if (random.random() < featureFrequency[f]):
      features.append(f)
      for k in range(numCategories):
        scores[k] += weights[f][k]

  multinomial = drawFromDirichlet(scores)

  num_samples = 20

  buckets = sampleFromMultinomial(multinomial, num_samples)

  training_set_file.write("\t".join(map(str, buckets)) + "\t" + "\t".join(map(str, features)))
  training_set_file.write("\n")

feature_list_file.close()
training_set_file.close()
weights_file.close()





