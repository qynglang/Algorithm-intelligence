from fcon_rep import train_fcon
import numpy as np
def run_f_ref(n,ans):
    EEE1=np.zeros((8,50))
    EEE2=np.zeros((8,50))
    EEE=np.zeros((8,50))
    Numm=np.zeros(50)
    for i in range (0,50):
        EE1,EE2,EE,Num=train_fcon(n,ans)
        EEE1[:,i]=EE1
        EEE2[:,i]=EE2
        EEE[:,i]=EE
        Numm[i]=Num
    np.save('Data/'+str(n)+'_EEE1.npy',EEE1)
    np.save('Data/'+str(n)+'_EEE2.npy',EEE2)
    np.save('Data/'+str(n)+'_EEE.npy',EEE)
    np.save('Data/'+str(n)+'_Numm.npy',Numm)
         
        
    