f1=open('weightsf2','wb')

with open('weights1', 'rb') as fi1:
    contents = fi1.read()
    f1.write(contents)

with open('weights2', 'rb') as fi2:
    contents = fi2.read()
    f1.write(contents)

    
f1.close()
