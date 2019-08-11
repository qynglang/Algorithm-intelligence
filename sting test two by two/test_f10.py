from fcon_2 import train_fcon
import numpy as np
#ac=np.zeros((50,2))
#acc=np.zeros((50,5))
#accc=np.zeros((20,8))
acccc=np.zeros((20,10))
for i in range (0,20):
    acccc[i,:]=train_fcon(10)
    if i%5==0:
        np.save('Data/acccc.npy',acccc)