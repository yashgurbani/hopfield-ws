#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yash
"""
import numpy as np
from random import choice

n = int(input('Enter # of neurons'))
m = int(input('Enter # of memory states'))
flip = round(0.25 * n)
MemoryMatrix = np.zeros((m, n))

for a in range(0, m):
    mem = []
    for i in range(0, n):
        x = choice([1, -1])
        mem.append(x)

    MemoryMatrix[a] = mem
    x = 0
print (MemoryMatrix)
Mt = MemoryMatrix.transpose()
print (Mt)
O = (1/n)*np.dot(MemoryMatrix,Mt)
print(O)
