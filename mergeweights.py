f1=open('weightsf2','w')
fi1=open('weights1','r')
fi2=open('weights2','r')

for line in fi1:
    f1.write(line)
for line in fi2:
    f1.write(line)

    
f1.close()
fi1.close()
fi2.close()
