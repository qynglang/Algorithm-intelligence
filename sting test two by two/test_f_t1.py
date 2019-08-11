from fcon_3 import train_fcon
import numpy as np
#ac=np.zeros((50,2))
acc=np.zeros((16,5))
#accc=np.zeros((16,8))
#acccc=np.zeros((50,10))
for i in range (0,16):
    #ac[i,:]=train_fcon(2)
    #acc[i,:]=train_fcon(5)
    acc[i,:]=train_fcon(5,1000)
    #acccc[i,:]=train_fcon(10)
    if i%5==0:
        #np.save('Data/ac.npy',ac)
        #np.save('Data/acc.npy',acc)
        np.save('Data/acc_t1.npy',acc)
        #np.save('Data/acccc.npy',acccc)