#!/bin/env python
# Testing the NPEET estimators

import entropy_estimators as ee
from math import log, pi
import numpy as np
import numpy.random as nr
import random
from numpy.linalg import det


def loadCSV_as_ndarray(data):
    myfile = np.genfromtxt(data, delimiter=',')
    return myfile


mydata = loadCSV_as_ndarray('MW1.csv')

x = mydata[:, 0]
y = mydata[:, 1]
z = mydata[:, 37]
h = np.ndarray(shape=(250,1), dtype=int, order='F')

for i in range(250):
    h[i, :] = random.randint(1, 2)
# x = [0, 0, 0, 0, 1, 1, 1, 1]
# y = [0, 0, 1, 1, 0, 0, 1, 1]
# z = [0, 1, 1, 0, 0, 1, 0, 1]
# h = [0, 0, 1, 0, 1, 1, 0, 1]
# print("H(x), H(y), H(z)", ee.entropyd(x), ee.entropyd(y), ee.entropyd(z))
# print("H(x:y), etc", ee.midd(x, y), ee.midd(z, y), ee.midd(x, z))
# print("H(x:y|z), etc", ee.cmidd(x, y, z), ee.cmidd(z, y, x), ee.cmidd(x, z, y))

print(ee.cmiddd(x, y, z))
print(ee.cmidd(x, y, z, h))
