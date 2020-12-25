import numpy as np
import time
import random

start_time = time.time()

a = []
b = []
c = []
B = np.random.rand(20,50)
for i in range(5000):
    A = np.random.rand(50,20)
    
    a.append(A)
    b.append(B)
    
    
    C = np.matmul(a[i],b[i])
    c.append(C)
    

    
print(time.time()-start_time)