#!/usr/bin/python

import sys
import csv
import math
import random

alpha = float(sys.argv[1])
beta = float(sys.argv[2])

for i in range(0, 1000):
	p = random.betavariate(alpha, beta)
	heads = 0
	tails = 0
	for j in range(0, 50):
		if random.random() < p: heads += 1
		else: tails += 1
	print str(heads) + "\t" + str(tails)