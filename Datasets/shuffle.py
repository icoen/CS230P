""" shuffles tweets """

import numpy as np
from math import floor
import random

np.random.seed(10)
"""
f=open('twtfulldemn.txt', 'rU')
f2=open('./trainvaltest/demfulltrain.txt','w')
f3=open('./trainvaltest/demfullval.txt','w')
f4=open('./trainvaltest/demfulltest.txt','w')
"""
f=open('twtfullrepn.txt', 'rU')
f2=open('./trainvaltest/repfulltrain.txt','w')
f3=open('./trainvaltest/repfullval.txt','w')
f4=open('./trainvaltest/repfulltest.txt','w')

x_all=list(f.readlines())
x_len=len(x_all)
print(x_len)
random.shuffle(x_all,random.random)

trainnum=int(floor(.8*x_len))
valnum=int(floor(.9*x_len))

def writelist(listn, fn):
    for el in listn:
        fn.write(el)

writelist(x_all[:trainnum],f2)
writelist(x_all[trainnum:valnum],f3)
writelist(x_all[valnum:],f4)
