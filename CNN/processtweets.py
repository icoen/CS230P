import fileinput
import re
import json
import math

f=open('twtrept.txt', 'rU')
f2=open('twtreptrain.txt','w')

for line in f:
    sumstring=re.sub('Republican,Rep.*?,', '', line)
    f2.write(sumstring)

f3=open('twtdemt.txt', 'rU')
f4=open('twtdemtrain.txt','w')

for line in f3:
    sumstring=re.sub('Democrat,Rep.*?,', '', line)
    f4.write(sumstring)

f.close()
f2.close()
f3.close()
f4.close()
