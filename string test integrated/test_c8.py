from con1 import train_cov
import numpy as np
#ac=np.zeros((20,2))
#acc=np.zeros((20,5))
accc=np.zeros((20,8))
# acccc=np.zeros((50,10))
for i in range (0,20):
    accc[i,:]=train_cov(8)
    if i%5==0:
        np.save('Data/accc.npy',accc)