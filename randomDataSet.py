import random
import numpy
f = open("random/dataset.csv", "w")

random.seed()

for row in range(1000):
    x = numpy.random.randn(100,10)
    s = ""
    for elem in x[0]:
        s += str(elem)
        s += ","
    s += str(random.randint(0,1))    
    print s
    f.write(s+"\n")
f.close()
    

