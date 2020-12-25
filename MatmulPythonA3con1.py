import numpy as np
import time
import random

start_time = time.time()

a = []
b = []
c = []

for i in range(100):
    A = np.random.rand(500,500)
    B = np.random.rand(500,400)
    a.append(A)
    b.append(B)
    
    
    C = np.matmul(a[i],b[i])
    c.append(C)
    

    
print(time.time()-start_time)