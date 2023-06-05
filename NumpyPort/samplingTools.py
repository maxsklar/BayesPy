
#!/usr/bin/python
#
# Sampling library from a variety of different distributions
#
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2012 Max Sklar,
#  2023 (numpy version) Jefkine Kafunah, Max Sklar

import numpy as np


def drawCategory(distribution):
    K = len(distribution)

    r = np.sum(distribution) * np.random.random()
    runningTotal = np.cumsum(distribution)
    k = np.searchsorted(runningTotal, r)

    if k < K:
        return k
    else:
        return K - 1
