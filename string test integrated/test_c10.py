from con1 import train_cov
import numpy as np
#ac=np.zeros((20,2))
#acc=np.zeros((20,5))
#accc=np.zeros((20,8))
acccc=np.zeros((20,10))
for i in range (0,20):
    acccc[i,:]=train_cov(10)
    if i%5==0:
        np.save('Data/acccc.npy',acccc)