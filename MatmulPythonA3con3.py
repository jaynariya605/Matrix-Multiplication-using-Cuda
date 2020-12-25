import numpy as np
import time
import random

start_time = time.time()

a = []
b = []
c = []
B = np.random.rand(4000,9)
for i in range(1000):
    A = np.random.rand(6,4000)
    
    a.append(A)
    b.append(B)
    
    
    C = np.matmul(a[i],b[i])
    c.append(C)
    

    
print(time.time()-start_time)