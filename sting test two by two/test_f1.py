from fcon_2 import train_fcon
import numpy as np
ac=np.zeros((50,2))
acc=np.zeros((50,5))
accc=np.zeros((50,8))
acccc=np.zeros((50,10))
for i in range (0,50):
    ac[i,:]=train_fcon(2)
    acc[i,:]=train_fcon(5)
    accc[i,:]=train_fcon(8)
    acccc[i,:]=train_fcon(10)
    if i%5==0:
        np.save('Data/ac_1.npy',ac)
        np.save('Data/acc_1.npy',acc)
        np.save('Data/accc_1.npy',accc)
        np.save('Data/acccc_1.npy',acccc)