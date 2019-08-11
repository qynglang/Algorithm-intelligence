import numpy as np
from data_gene import forward
def foward_error (n,ans):
    data=np.load('data_4.npy')
    E_min=np.zeros((25))
    EEE=np.zeros((6,25))
    for i in range (0,5):
        N_str='3_c_f'+str(i)
        w1=np.load('Data/'+str(N_str)+'weight1_t.npy')
        w2=np.load('Data/'+str(N_str)+'weight2_t.npy')
        wf1=np.load('Data/'+str(N_str)+'weightf_t.npy')
        bf1=np.load('Data/'+str(N_str)+'biasf_t.npy')

        O2=np.zeros((6,1024))
        E=np.zeros(6)
        EE=np.zeros((6,w1.shape[4]))

        for j in range (0,w1.shape[4]):
            II=forward(data[:,:,:,n],w1[:,:,:,:,j],w2[:,:,:,:,j])
            error01=II[1,:]-II[0,:]
            error02=II[2,:]-II[0,:]
            error23=np.zeros((6,180))
            error13=np.zeros((6,180))
            for i in range (1,7):
                error23[i-1,:]=II[2+i,:]-II[2,:]
            for i in range (1,7):
                error13[i-1,:]=II[2+i,:]-II[1,:]

            O1=np.dot(error01.reshape(1,180),wf1[:,:,j])+bf1[:,j]
            for i in range (0,6):
                O2[i,:]=np.dot(error23[i,:].reshape(1,180),wf1[:,:,j])+bf1[:,j]

            for i in range (0,6):
                E[i]=abs(O2[i,:]-O1).sum()
            EE[:,j]=E
          


        
            E_min_n=np.zeros((5))
            for i in range (0,w1.shape[4]):
                E_min_n[i]=np.argmin(EE[:,i])
        E_min[i:i+5]=E_min_n
        EEE[:,i:i+5]=EE
    #print(EE)
    print(E_min[E_min==ans].size)
    #print(E1_min_n[E1_min_n==ans].size)
    print(np.average(EEE,axis=1))
    print(np.argsort(np.average(EEE,axis=1)))
    #print(np.average(E2,axis=0))
    print(np.average(EEE,axis=1)[ans]/EEE.mean())
    
    